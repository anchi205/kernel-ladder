"""Profiler feedback without ncu: per-kernel GPU time from torch.profiler (CUPTI activity API),
plus achieved memory bandwidth against a measured peak.

ncu and compute-sanitizer do not support this GPU (sm_121) in the installed CUDA 12.8 toolkit,
so these are deliberate substitutes, not equivalents: no occupancy or cache counters.
"""
import functools

import torch
from torch.profiler import ProfilerActivity, profile


@functools.lru_cache(maxsize=1)
def peak_bandwidth_gbs() -> float:
    """Best observed memory bandwidth: the larger of a 1 GiB streaming read (torch.sum) and a copy (read + write counted).

    A copy alone under-reports the ceiling on this GPU (246 vs 264 GB/s measured; a small 256 MiB copy gave 228), which made
    kernels near the ceiling show up as 'over 100% of peak'. The streaming read is the right ceiling for read-dominated ops.
    """
    x = torch.empty(1 << 28, dtype=torch.float32, device="cuda").normal_()
    dst = torch.empty_like(x)

    def best(fn, bytes_moved):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        top = 0.0
        for _ in range(10):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(); fn(); e1.record(); e1.synchronize()
            top = max(top, bytes_moved / (e0.elapsed_time(e1) / 1e3) / 1e9)
        return top

    peak = max(best(lambda: dst.copy_(x), 2 * x.numel() * 4), best(lambda: x.sum(), x.numel() * 4))
    del x, dst
    torch.cuda.empty_cache()
    return peak


def profile_call(fn, inputs, bytes_moved: int, iters: int = 5) -> dict:
    for _ in range(3):
        fn(*inputs)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn(*inputs)
        torch.cuda.synchronize()
    kernels = {}
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA:
            t = kernels.setdefault(ev.name, [0.0, 0])
            t[0] += ev.device_time_total
            t[1] += 1
    per_call_us = sum(t for t, _ in kernels.values()) / iters
    gbs = bytes_moved / (per_call_us * 1e-6) / 1e9 if per_call_us else 0.0
    peak = peak_bandwidth_gbs()
    top = sorted(kernels.items(), key=lambda kv: -kv[1][0])[:4]
    return {
        "kernels_per_call": round(sum(c for _, c in kernels.values()) / iters, 2),
        "gpu_us_per_call": round(per_call_us, 1),
        "achieved_gbs": round(gbs, 1),
        "peak_gbs": round(peak, 1),
        "pct_of_peak": round(100 * gbs / peak, 1),
        "top_kernels": [f"{n[:60]} x{c // iters}: {t / iters:.0f}us" for n, (t, c) in top],
    }
