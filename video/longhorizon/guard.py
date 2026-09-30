"""Resource guard: caps BLAS/torch threads, optionally pins CPUs, sets CUDA_VISIBLE_DEVICES and starts an RSS watchdog that exits the
process if resident memory exceeds a limit. Environment variables (optional) override the arguments of install():
LH_THREADS (thread cap), LH_CPUS (CPU range 'a-b' for affinity), LH_RSS_GB (RSS limit in GB). Inside a process the assigned GPU is cuda:0."""
import os, sys, threading, time

def _rss_gb():
    try:
        with open("/proc/self/status") as f:
            for ln in f:
                if ln.startswith("VmRSS:"): return int(ln.split()[1]) / 1048576
    except Exception: pass
    return 0.0
def cgroup_mem():
    """anon memory (the RAM a run actually owns), not memory.current (which also counts page cache -- e.g. 51 GiB of model files
    today -- and would trip the 85% rule on a healthy process). Falls back to memory.current if memory.stat has no anon line."""
    try:
        cur = None
        for ln in open("/sys/fs/cgroup/memory.stat"):
            if ln.startswith("anon "): cur = int(ln.split()[1]); break
        if cur is None: cur = int(open("/sys/fs/cgroup/memory.current").read())
        mx = open("/sys/fs/cgroup/memory.max").read().strip()
        return cur / 2**30, (float("inf") if mx == "max" else int(mx) / 2**30)
    except Exception: return 0.0, float("inf")


def install(rss_gb=None, threads=None, cpus=None, gpu=None, period=3.0):
    """Call before importing torch. Sets thread env vars, CPU affinity, CUDA_VISIBLE_DEVICES and starts an RSS watchdog that hard-exits.
    Precedence: an inherited CUDA_VISIBLE_DEVICES (even "" = CPU) beats `gpu`; LH_CPUS beats `cpus`; LH_THREADS / LH_RSS_GB cap `threads` /
    `rss_gb`. Defaults when neither is given: GPU 0, 24 threads, no CPU pinning, 55 GB."""
    env = os.environ
    if "CUDA_VISIBLE_DEVICES" in env:
        if gpu is not None and str(gpu) != env["CUDA_VISIBLE_DEVICES"]:
            print(f"guard: keeping inherited CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']!r} (install asked gpu={gpu!r})")
        gpu = env["CUDA_VISIBLE_DEVICES"]
    else:
        gpu = "0" if gpu is None else str(gpu)
        env["CUDA_VISIBLE_DEVICES"] = gpu
    cap = lambda arg, var, dflt, typ: min(typ(arg), typ(env[var])) if (arg is not None and env.get(var)) else \
        typ(env[var]) if env.get(var) else (typ(arg) if arg is not None else dflt)
    threads = cap(threads, "LH_THREADS", 24, int)
    rss_gb = cap(rss_gb, "LH_RSS_GB", 55.0, float)
    cpus = env.get("LH_CPUS") or cpus
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ[v] = str(threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"; os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    if cpus:
      try:
        a, b = cpus.split("-"); os.sched_setaffinity(0, set(range(int(a), int(b) + 1)))
      except Exception as e: print("guard: affinity not set:", e)
    if "torch" in sys.modules:
        import torch; torch.set_num_threads(threads)
    state = dict(peak=0.0)
    def watch():
        while True:
            r = _rss_gb(); state["peak"] = max(state["peak"], r); cur, mx = cgroup_mem()
            if r > rss_gb or cur > 0.97 * mx:
                print(f"\n!!! guard: RSS {r:.1f} GB (limit {rss_gb}) / cgroup {cur:.1f}/{mx:.0f} GB -> exiting", flush=True)
                os._exit(3)
            time.sleep(period)
    threading.Thread(target=watch, daemon=True).start()
    cur, mx = cgroup_mem()
    print(f"guard: gpu={gpu} threads={threads} cpus={cpus} rss_limit={rss_gb}GB | cgroup mem {cur:.1f}/{mx:.0f} GB")
    return state
