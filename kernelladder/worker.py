import dataclasses
import json
import os
import sys
import threading
import time

from .verify import verify

# Defaults sized for a 128 GiB unified-memory box; the benchmark shapes need well under 1 GiB.
MAX_GPU_FRAC = float(os.environ.get("KL_MAX_GPU_FRAC", "0.25"))
MAX_RAM_GB = float(os.environ.get("KL_MAX_RAM_GB", "16"))


def _rss_gb() -> float:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**30


def _watch_ram(limit_gb: float, period: float = 0.05):
    """Kill the worker when resident memory passes the cap.

    RLIMIT_AS is unusable here: CUDA reserves far more address space than it uses.
    The runner turns the nonzero exit and this stderr line into a failed report.
    """
    def loop():
        while True:
            rss = _rss_gb()
            if rss > limit_gb:
                os.write(2, f"killed: resident memory {rss:.1f} GB exceeded cap {limit_gb:g} GB\n".encode())
                os._exit(137)
            time.sleep(period)
    threading.Thread(target=loop, daemon=True).start()


def _cap_gpu(frac: float):
    import torch
    if torch.cuda.is_available():
        # An allocation past the cap raises torch.OutOfMemoryError inside the candidate,
        # which verify() reports as a correctness failure. Covers torch allocations only.
        torch.cuda.set_per_process_memory_fraction(frac)


if __name__ == "__main__":
    args = sys.argv[1:]
    _watch_ram(MAX_RAM_GB)
    _cap_gpu(MAX_GPU_FRAC)
    if args[0] == "compare":
        from .compare import compare
        from .verify import load_candidate
        print(json.dumps(compare(args[1], load_candidate(args[2])), default=str))
    else:
        import torch
        op, path, min_speedup = args[0], args[1], float(args[2])
        # "wide": the gate's sweep. Fresh inputs and an extra dtype, on top of the normal checks.
        kw = dict(dtypes=(torch.float32, torch.float16, torch.bfloat16), seed=12345) if "wide" in args[3:] else {}
        r = verify(op, path, min_speedup=min_speedup, **kw)
        print(json.dumps(dataclasses.asdict(r), default=str))
