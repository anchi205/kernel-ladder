"""Honest baselines: the reference is the correctness oracle, not the thing a hiring manager asks about.

`compare` times a candidate against the reference and against the strongest library alternatives
that exist for the op (PyTorch's fused attention, torch.compile, F.rms_norm, torch._int_mm).
Baselines that fail to build or run are reported as errors, never silently dropped.
"""
import torch

from .ops import OPS
from .verify import _time


def _compiled(op):
    return lambda: torch.compile(op.reference)


def _sdpa():
    import torch.nn.functional as F
    return lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True)


def _rms_norm():
    import torch.nn.functional as F
    return lambda x, w: F.rms_norm(x, (x.shape[-1],), w, 1e-6)


def _int_mm(a, b, sa, sb):
    return (torch._int_mm(a, b).float() * sa.float()[:, None] * sb.float()[None, :]).to(sa.dtype)


EXTRA = {
    "attention": {"torch SDPA (fused)": _sdpa},
    "rmsnorm": {"F.rms_norm": _rms_norm},
    "int8_matmul": {"torch._int_mm": lambda: _int_mm},
}


def baselines(op_name: str) -> dict:
    op = OPS[op_name]
    out = {"torch.compile(reference)": _compiled(op)}
    out.update(EXTRA.get(op_name, {}))
    return out


def compare(op_name: str, candidate_fn, dtypes=(torch.float32, torch.float16)) -> list[dict]:
    op = OPS[op_name]
    rows = []
    g = torch.Generator(device="cuda").manual_seed(0)
    for dtype in dtypes:
        inputs = op.make_inputs(op.bench_shape, dtype, "cuda", g)
        contenders = {"reference": lambda: op.reference, "candidate": lambda: candidate_fn}
        contenders.update(baselines(op_name))
        timings, errors = {}, {}
        for name, build in contenders.items():
            try:
                timings[name] = _time(build(), inputs, "cuda") * 1e3
            except Exception as e:
                errors[name] = f"{type(e).__name__}: {str(e)[:120]}"
        rows.append({"dtype": str(dtype).replace("torch.", ""), "ms": timings, "errors": errors})
    return rows


def format_rows(rows: list[dict]) -> str:
    lines = []
    for r in rows:
        ms, cand = r["ms"], r["ms"].get("candidate")
        lines.append(f"\n[{r['dtype']}]  {'contender':<28}{'ms':>9}   candidate speedup vs it")
        for name, t in ms.items():
            rel = "" if name == "candidate" or not cand else f"{t / cand:>8.2f}x"
            lines.append(f"      {name:<28}{t:>9.3f}   {rel}")
        for name, err in r["errors"].items():
            lines.append(f"      {name:<28}  ERROR {err}")
    return "\n".join(lines)
