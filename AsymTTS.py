import numpy as np, scipy.sparse as sp, torch, time, warnings, os
warnings.filterwarnings("ignore")                      # silence sparse/efficiency warnings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU if present
DT = torch.float32                                     # compute precision

# ---- seeding policy --------------------------------------------------------
NUM_SEEDS    = 10      # how many seeds (= independent batches) per run
RANDOM_SEEDS = True    # True: fresh OS-entropy seeds each run (independent samples)
                       # False: fixed list range(NUM_SEEDS) (fully reproducible)

def make_seeds(n=NUM_SEEDS, randomize=RANDOM_SEEDS):
    # Produce the list of seeds for one run and print it (so the run can be reproduced later).
    if randomize:
        seeds = np.random.default_rng().integers(0, 2**31-1, size=n).tolist()  # fresh entropy each call
    else:
        seeds = list(range(n))                          # deterministic fallback
    print(f" seeds ({'random' if randomize else 'fixed'}): {seeds}")
    return seeds
# ----------------------------------------------------------------------------

CONFIG = {
    # paper Table 1 reference scores (MIS as-is).  bk=None means unknown ("-").
    "instances": [
        #{"name": "C1000.9",     "path": "C1000.9.clq.txt",     "cim": 6,  "osa": 6,  "bk": 6},
        #{"name": "C2000.5",     "path": "C2000.5.clq.txt",     "cim": 15, "osa": 16, "bk": 17},
        #{"name": "C2000.9",     "path": "C2000.9.clq.txt",     "cim": 6,  "osa": 6,  "bk": 6},
        #{"name": "C4000.5",     "path": "C4000.5.clq.txt",     "cim": 16, "osa": 17, "bk": 18},
        #{"name": "p_hat1500-2",     "path": "p_hat1500-2.clq.txt",     "cim": 58, "osa": 62, "bk": 62},
        #{"name": "p_hat1500-1",     "path": "p_hat1500-1.clq.txt",     "cim": 77, "osa": 87, "bk": 87},
        #{"name": "DSJC1000_5",     "path": "DSJC1000_5.clq.txt",     "cim": 14, "osa": 15, "bk": 15},
        #{"name": "p_hat1500-3",     "path": "p_hat1500-3.clq.txt",     "cim": 11, "osa": 11, "bk": 12}
    ],
    "methods": ["asymmetric"],         

    "B": 100, "steps": 300, "seeds": None,             # batch size, annealing steps; seeds filled per run
    "dt": 0.1, "a0": 1.0, "noise": 0.05,               # time step, pump target a0, symmetry-breaking noise std
    "Delta": 1.0, "c0_relu": 0.7, "P_min": 0.5, "P_max": 2.5, "P_exp": 3, "init_amp": 0.0,
    #   Delta   = reward field (pulls every unit toward "selected")
    #   c0_relu = coupling gain on the force
    #   P_min/P_max/P_exp = penalty weight ramp  P(step) = P_min + (P_max-P_min)*(step/steps)^P_exp
    #   init_amp = amplitude of random init (0.0 = start exactly at the silent point x=0)
}

# ---------------- I/O ----------------
def load_dimacs(path):
    # Parse a DIMACS .clq edge-list file into a dense symmetric {0,1} adjacency matrix.
    N = None; E = []
    with open(path) as f:
        for line in f:
            t = line.split()
            if not t or t[0] == "c": continue          # skip blank/comment lines
            if t[0] == "p": N = int(t[2])              # 'p' line carries the node count
            elif t[0] == "e":                          # 'e u v' line is an edge (1-indexed)
                u, v = int(t[1])-1, int(t[2])-1        # convert to 0-indexed
                if u != v: E.append((u, v))            # ignore self-loops
    A = np.zeros((N, N), np.float32)
    for u, v in E: A[u, v] = 1; A[v, u] = 1            # symmetric fill (undirected graph)
    return A

def to_sp(A):
    # Convert the dense adjacency to a torch sparse-COO tensor on the device (for sparse mm).
    c = sp.coo_matrix(A)                                                       # dense -> COO
    idx = torch.tensor(np.vstack([c.row, c.col]), dtype=torch.long, device=DEVICE)  # 2xNNZ index matrix
    return torch.sparse_coo_tensor(idx, torch.tensor(c.data,dtype=DT,device=DEVICE), A.shape).coalesce()

# ---------------- asymmetric (your method) ----------------
def asymmetric(A_sp, N, cfg, seed):
    # Asymmetric ReLU-coupling Simulated Bifurcation for MIS, run as B parallel trajectories.
    # Only ACTIVE units (x>0) exert repulsion (ReLU coupling), so the artificial O(deg) field at
    # the neutral point disappears; symmetry-breaking noise escapes the empty set.
    B, steps, dt = cfg["B"], cfg["steps"], cfg["dt"]                    # batch, # steps, time step
    a0, noise, c0, D = cfg["a0"], cfg["noise"], cfg["c0_relu"], cfg["Delta"]  # pump, noise std, gain, reward
    Pmin, Pmax, Pexp, amp = cfg["P_min"], cfg["P_max"], cfg["P_exp"], cfg["init_amp"]  # penalty ramp + init amp
    g = torch.Generator(device=DEVICE).manual_seed(int(seed))          # per-batch RNG (seed controls the draws)
    x = (amp*(2*torch.rand((B,N),generator=g,dtype=DT,device=DEVICE)-1) if amp>0
         else torch.zeros((B,N),dtype=DT,device=DEVICE))               # positions: start at silent point x=0
    y = torch.zeros((B,N),dtype=DT,device=DEVICE); sq = noise*dt**0.5  # momenta start at 0; sqrt(dt) noise scale
    for step in range(steps):                                          # annealing loop
        fr = step/steps; a = a0*fr; P = Pmin+(Pmax-Pmin)*fr**Pexp       # ramp pump a:0->a0 and penalty P:min->max
        F = D - P*torch.sparse.mm(A_sp, torch.clamp(x,min=0.0).t()).t()  # force = reward - penalty*(active-neighbour sum)
        y = y + (-(a0-a)*x + c0*F)*dt + sq*torch.randn((B,N),generator=g,dtype=DT,device=DEVICE)  # momentum: bifurcation + force + noise
        x = x + a0*y*dt                                                # position update (symplectic Euler)
        w = x.abs()>=1; x = torch.clamp(x,-1,1); y = torch.where(w,torch.zeros_like(y),y)  # inelastic walls at +/-1
    return (torch.sign(x)+1)/2                                         # binarise sign(x) in {-1,1} -> {0,1}

def valid_sizes(z, A_sp):
    """Per-run feasible IS size for the whole batch (0 if infeasible). Length-B numpy array."""
    Az = torch.sparse.mm(A_sp, z.t()).t()        # per node, number of selected neighbours
    viol = (z*Az).sum(1)/2.0                      # # edges inside the selected set (= constraint violations)
    sizes = z.sum(1)                              # # selected nodes
    vs = torch.where(viol < 0.5, sizes, torch.zeros_like(sizes))  # keep size only if feasible (0 violations)
    return vs.cpu().numpy().astype(int)

# ---------------- score table (best IS + success rate) ----------------
def run(cfg):
    # Print a table of best feasible IS and success rate (% of runs reaching that best) per instance.
    seeds = make_seeds()                                               # fresh seeds for this run (printed)
    cfg = dict(cfg); cfg["seeds"] = seeds
    print("="*86)
    print(f" Asymmetric MIS suite   device={DEVICE}   methods={cfg['methods']}  B={cfg['B']} steps={cfg['steps']} n_seeds={len(seeds)}")
    print("="*86)
    hdr = f" {'instance':12s} {'n':>5s} {'dens':>5s} | "                    # build the header row
    for m in cfg["methods"]: hdr += f"{m[:9]:>9s} {'succ%':>6s} "
    hdr += f"| {'CIM':>4s} {'OSA':>4s} {'BK':>4s}  gap(BK)"
    print(hdr); print("-"*86)
    for inst in cfg["instances"]:
        if not os.path.exists(inst["path"]):                               # skip missing files gracefully
            print(f" {inst['name']:12s}  -- file '{inst['path']}' not found, skipped"); continue
        A = load_dimacs(inst["path"]); N = A.shape[0]; A_sp = to_sp(A)      # load graph + sparse form
        dens = A.sum()/(N*(N-1))                                           # edge density
        row = f" {inst['name']:12s} {N:5d} {dens:5.2f} | "; ours_best = None
        for meth in cfg["methods"]:                                        # (only "asymmetric" in this build)
            pool = []
            for sd in cfg["seeds"]:
                if meth == "asymmetric":                                   # gather feasible IS sizes per seed
                    pool.append(valid_sizes(asymmetric(A_sp,N,cfg,sd), A_sp))
            pool = np.concatenate([np.atleast_1d(p) for p in pool])        # flatten across seeds*batch
            b = int(pool.max()) if pool.size else 0                        # best IS found
            succ = 100.0*float((pool == b).mean()) if pool.size else 0.0   # % of runs hitting that best
            row += f"{b:9d} {succ:5.1f}% "
            if meth == "asymmetric": ours_best = b
        bk = inst["bk"]; bks = "-" if bk is None else str(bk)              # best-known (or '-')
        gap = "" if (bk is None or ours_best is None) else f"{ours_best-bk:+d}"  # gap to best-known
        row += f"| {inst['cim']:>4d} {inst['osa']:>4d} {bks:>4s}  {gap}"
        print(row)
    print("="*86)

def dump_selected(path, cfg, seeds, out="selected.txt"):
    """Run the solver; print & save the selected VERTICES of the best feasible trajectory (1-indexed)."""
    A = load_dimacs(path); N = A.shape[0]; A_sp = to_sp(A)
    best_size = -1; best_sel = None; best_seed = None
    for sd in seeds:
        z = asymmetric(A_sp, N, cfg, int(sd))
        sizes = valid_sizes(z, A_sp)
        j = int(sizes.argmax())
        if int(sizes[j]) > best_size:
            best_size = int(sizes[j])
            best_sel  = np.where(z[j].detach().cpu().numpy() > 0.5)[0]
            best_seed = int(sd)
    verts = sorted(int(v) + 1 for v in best_sel)              # +1 -> 1-indexed, like the .clq file
    print(f"{path}: best feasible IS = {best_size}  (seed {best_seed})")
    print(f"selected vertices (1-indexed, n={len(verts)}):")
    print(verts)
    with open(out, "w") as f:
        f.write(" ".join(map(str, verts)) + "\n")
    print(f"saved to {out}")
    return verts

# ---------------- TTS measurement ----------------
def measure_tts(path, target, B=2000, seeds=None, p_t=0.99, cfg=None):
    """TTS for the asymmetric solver. success = (IS >= target).
    TTS = N_traj * T_batch/B ,  N_traj = ln(1-p_t)/ln(1-p).
    seeds=None -> draw fresh random seeds for this run (independent sample)."""
    cfg = dict(CONFIG if cfg is None else cfg); cfg["B"] = B               # work on a copy; override batch
    if seeds is None: seeds = make_seeds()                                 # fresh seeds for this run (printed)
    if not os.path.exists(path):
        print(f" file '{path}' not found"); return None
    A = load_dimacs(path); N = A.shape[0]; A_sp = to_sp(A)                 # load graph + sparse form
    _ = asymmetric(A_sp, N, dict(cfg, B=8), 999)                          # warm-up run (compile/caches; NOT timed)
    if DEVICE.type == "cuda": torch.cuda.synchronize()                    # make sure GPU is idle before timing
    succ = 0; tot = 0; per_seed = []; times = []
    for sd in seeds:                                                       # each seed = one full batch of B trajectories
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        t0 = time.time()
        z = asymmetric(A_sp, N, cfg, int(sd))                             # run the batch
        if DEVICE.type == "cuda": torch.cuda.synchronize()                # wait for completion before stopping clock
        times.append(time.time() - t0)
        sizes = valid_sizes(z, A_sp)
        hits = int((sizes >= target).sum()); succ += hits; tot += B; per_seed.append(hits / B)  # count successes
    p = succ / tot; T_batch = float(np.median(times))                     # per-trajectory success prob; median batch time
    print(f" target>={target}:  {succ}/{tot} successes  ->  p = {p:.4%}")
    print(f" per-seed p: {['%.2f%%'%(x*100) for x in per_seed]}")
    print(f" T_batch(B={B}) = {T_batch:.4f}s   (amortized {T_batch/B*1e3:.4f} ms / trajectory)")
    if succ == 0:                                                          # target never reached -> TTS undefined
        print(f" p = 0  ->  TTS = INF  (target not reached in {tot} runs; lower bound only)")
        return float("inf")
    N_traj = 1.0 if p >= 1.0 else np.log(1-p_t)/np.log(1-p)                # # trajectories for prob p_t (guard p=1)
    TTS = N_traj * (T_batch / B)                                           # time-to-solution (amortized over the batch)
    print(f" N_traj(@{p_t:.0%} conf) = {N_traj:.1f} trajectories")
    print(f" ==> TTS = {TTS:.4f} s")
    return TTS

# ---------------- main ----------------
if __name__ == "__main__":
    # 1) score table (best IS + success%) -- draws fresh seeds, prints them
    run(CONFIG)

    #2) TTS  (edit path/target/steps as needed); seeds drawn fresh each run and printed
    measure_tts("p_hat1500-3.clq.txt", target=11, B=2000)      # seeds=None -> 10 fresh random seeds

    #cfg = dict(CONFIG, B=2000, steps=500)
    
    #dump_selected("DSJC1000_5.clq.txt", cfg, seeds=[489192590], out="phat1.txt")

    #A = load_dimacs("p_hat1500-1.clq.txt")
    #print(A.shape[0], int(A.sum()//2))
