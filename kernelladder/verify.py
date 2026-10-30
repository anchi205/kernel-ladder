"""The check: correctness against the reference, then timing against the baseline.

A check that cannot fail proves nothing, so every failure carries an
EXPECTED / SAW / FENCE / DEAD ENDS report the loop can feed back to the agent.
"""
import importlib.util
import statistics
import time
from dataclasses import dataclass, field

import torch

from .ops import OPS


@dataclass
class Report:
    green: bool
    stage: str                      # load | correctness | speed | ok
    expected: str = ""
    saw: str = ""
    fence: str = "candidate file only"
    speedup: float | None = None
    detail: dict = field(default_factory=dict)

    def profile_lines(self) -> list[str]:
        c, r = self.detail.get("profile"), self.detail.get("ref_profile")
        if not c:
            return []
        def line(tag, p):
            return (f"PROFILE    {tag}: {p['kernels_per_call']:g} kernel(s)/call, {p['gpu_us_per_call']:.0f}us, "
                    f"{p['achieved_gbs']:.0f} GB/s = {p['pct_of_peak']:.0f}% of peak ({p['peak_gbs']:.0f} GB/s)")
        out = [line("candidate", c)]
        if r:
            out.append(line("reference", r))
        out.append("KERNELS    " + "; ".join(c["top_kernels"]))
        return out

    def feedback(self, dead_ends: list[str] | None = None) -> str:
        if self.green:
            return "\n".join([f"GREEN  speedup {self.speedup:.2f}x"] + self.profile_lines())
        return "\n".join([
            f"FAILED      {self.stage}",
            f"EXPECTED    {self.expected}",
            f"SAW         {self.saw}",
            f"FENCE       {self.fence}",
            f"DEAD ENDS   {', '.join(dead_ends) if dead_ends else 'none'}",
        ] + self.profile_lines())


def load_candidate(path: str):
    spec = importlib.util.spec_from_file_location("candidate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not callable(getattr(mod, "forward", None)):
        raise AttributeError("candidate must define forward(*inputs)")
    return mod.forward


def _as_tuple(x):
    return tuple(x) if isinstance(x, (tuple, list)) else (x,)


def _mismatch(got, want, rtol, atol, int_atol):
    """Return a description of how `got` differs from `want`, or None if it matches."""
    got, want = _as_tuple(got), _as_tuple(want)
    if len(got) != len(want):
        return f"{len(got)} output(s), reference returns {len(want)}"
    for k, (g_, w_) in enumerate(zip(got, want)):
        if not isinstance(g_, torch.Tensor):
            return f"output {k} is {type(g_).__name__}, not a tensor"
        if g_.shape != w_.shape or g_.dtype != w_.dtype:
            return (f"output {k}: shape {tuple(g_.shape)} dtype {g_.dtype}, "
                    f"reference has shape {tuple(w_.shape)} dtype {w_.dtype}")
        if w_.is_floating_point():
            if torch.allclose(g_, w_, rtol=rtol, atol=atol):
                continue
            err = (g_.double() - w_.double()).abs()
            i = int(err.argmax())
            return (f"output {k}: max abs err {err.max().item():.3e} at flat index {i}: "
                    f"got {g_.flatten()[i].item():.6g}, want {w_.flatten()[i].item():.6g}")
        err = (g_.long() - w_.long()).abs()
        if err.numel() and err.max().item() > int_atol:
            i = int(err.argmax())
            return (f"output {k}: integer diff {err.max().item()} (allowed {int_atol}) at flat index {i}: "
                    f"got {g_.flatten()[i].item()}, want {w_.flatten()[i].item()}")
    return None


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def _time(fn, inputs, device, warmup=5, iters=20) -> float:
    for _ in range(warmup):
        fn(*inputs)
    _sync(device)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(*inputs)
        _sync(device)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def verify(op_name: str, candidate_path: str, device: str | None = None,
           dtypes=(torch.float32, torch.float16), min_speedup: float = 1.0,
           seed: int = 0) -> Report:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    op = OPS[op_name]
    source = open(candidate_path).read()
    hits = [b for b in op.banned if b in source]
    if hits:
        return Report(False, "fence", "candidate implements the op itself",
                      f"source contains banned name(s): {', '.join(hits)}",
                      fence="no calls to the op's own library function, no kernelladder imports")
    try:
        fn = load_candidate(candidate_path)
    except Exception as e:
        return Report(False, "load", "candidate imports and defines forward()", repr(e))

    g = torch.Generator(device=device).manual_seed(seed)
    for dtype in dtypes:
        if device == "cpu" and dtype == torch.float16:
            continue  # CPU half math is not a meaningful reference
        rtol, atol = op.tol[dtype]
        for shape in op.shapes:
            inputs = op.make_inputs(shape, dtype, device, g)
            want = op.reference(*inputs)
            snapshot = [t.clone() for t in inputs]
            try:
                got = fn(*inputs)
            except Exception as e:
                return Report(False, "correctness", f"forward() runs on {shape} {dtype}", repr(e))
            if not all(torch.equal(a, b) for a, b in zip(inputs, snapshot)):
                return Report(False, "correctness", "forward() leaves its inputs unmodified",
                              f"an input tensor was changed in place on {shape} {dtype}")
            saw = _mismatch(got, want, rtol, atol, op.int_atol)
            if saw:
                return Report(False, "correctness", f"within rtol={rtol} atol={atol} on {shape} {dtype}", saw,
                              detail={"shape": shape, "dtype": str(dtype)})

    inputs = op.make_inputs(op.bench_shape, torch.float32, device, g)

    # Stand-in for racecheck (unsupported on this GPU): identical inputs must give bit-identical
    # outputs run after run. Races and unordered atomics show up as run-to-run differences.
    first = tuple(t.clone() for t in _as_tuple(fn(*inputs)))
    for _ in range(4):
        again = _as_tuple(fn(*inputs))
        for a, b in zip(first, again):
            if not torch.equal(a, b):
                n = int((a != b).sum())
                return Report(False, "determinism", "bit-identical output on repeated runs of the same input",
                              f"{n} of {a.numel()} elements differed between runs (race or unordered atomics?)")

    base = _time(op.reference, inputs, device)
    cand = _time(fn, inputs, device)
    speedup = base / cand
    detail = {"cand_ms": cand * 1e3, "base_ms": base * 1e3}
    if device == "cuda":
        try:
            from .profile import profile_call
            nbytes = (op.traffic_bytes(inputs) if op.traffic_bytes
                      else op.rw_factor * inputs[0].numel() * inputs[0].element_size())
            detail["profile"] = profile_call(fn, inputs, nbytes)
            detail["ref_profile"] = profile_call(op.reference, inputs, nbytes)
        except Exception as e:      # profiling is advice, never a reason to fail a correct kernel
            detail["profile_error"] = repr(e)
    if speedup < min_speedup:
        return Report(False, "speed", f"speedup >= {min_speedup:.2f}x vs reference",
                      f"{speedup:.2f}x ({cand*1e3:.3f} ms vs {base*1e3:.3f} ms)",
                      speedup=speedup, detail=detail)
    return Report(True, "ok", speedup=speedup, detail=detail)
