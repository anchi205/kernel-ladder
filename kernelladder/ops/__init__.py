"""Reference ops: the fixed ground truth the verifier checks candidates against.

The agent may never edit this package; it is the read-only side of the loop.
"""
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class Op:
    name: str
    reference: Callable[..., torch.Tensor]
    make_inputs: Callable[[tuple, torch.dtype, str, torch.Generator], tuple]
    shapes: tuple          # correctness shapes, includes edge cases
    bench_shape: tuple     # one large shape for timing
    tol: dict              # dtype -> (rtol, atol)
    rw_factor: int = 2     # array-sizes of memory traffic at the roofline (read once + write once)
    int_atol: int = 1      # allowed difference for integer outputs (rounding ties)
    traffic_bytes: Callable | None = None   # minimum bytes moved, for the bandwidth roofline
    banned: tuple = ()     # substrings a candidate may not contain: calling the op itself is not a kernel


def _softmax(x):
    return torch.softmax(x, dim=-1)


def _rmsnorm(x, w, eps: float = 1e-6):
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (out * w.float()).to(x.dtype)


def _softmax_inputs(shape, dtype, device, g):
    x = torch.randn(shape, dtype=dtype, device=device, generator=g) * 8
    # softmax is shift-invariant, so a +100 offset on alternate rows leaves the answer
    # unchanged but overflows any implementation that skips max-subtraction
    if dtype == torch.float32:
        x[::2] += 100
    return (x,)


def _rmsnorm_inputs(shape, dtype, device, g):
    x = torch.randn(shape, dtype=dtype, device=device, generator=g)
    w = torch.randn(shape[-1], dtype=dtype, device=device, generator=g)
    return (x, w)


def _attention(q, k, v):
    """Causal scaled dot-product attention, naive: materializes the full score matrix."""
    s = (q @ k.transpose(-1, -2)) * (q.shape[-1] ** -0.5)
    keep = torch.ones(s.shape[-2:], dtype=torch.bool, device=q.device).tril()
    p = torch.softmax(s.masked_fill(~keep, float("-inf")).float(), dim=-1).to(q.dtype)
    return p @ v


def _attention_inputs(shape, dtype, device, g):
    return tuple(torch.randn(shape, dtype=dtype, device=device, generator=g) for _ in range(3))


def _int8_matmul(a, b, sa, sb):
    """Per-row / per-column scaled int8 matmul. Exact in fp32 for K <= 1024 (127*127*K < 2^24)."""
    acc = a.float() @ b.float()
    return (acc * sa.float()[:, None] * sb.float()[None, :]).to(sa.dtype)


def _int8_inputs(shape, dtype, device, g):
    m, k, n = shape
    a = torch.randint(-127, 128, (m, k), dtype=torch.int8, device=device, generator=g)
    b = torch.randint(-127, 128, (k, n), dtype=torch.int8, device=device, generator=g)
    sa = torch.rand(m, dtype=dtype, device=device, generator=g) * 0.02 + 0.001
    sb = torch.rand(n, dtype=dtype, device=device, generator=g) * 0.02 + 0.001
    return a, b, sa, sb


def _kv_quant(x):
    """Quantize new K/V entries to int8 with one absmax scale per (batch, head, token) row."""
    xf = x.float()
    scale = xf.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 127
    q = torch.round(xf / scale).clamp(-127, 127).to(torch.int8)
    return q, scale.squeeze(-1)


def _kv_inputs(shape, dtype, device, g):
    return (torch.randn(shape, dtype=dtype, device=device, generator=g) * 3,)


TOL = {torch.float32: (1e-4, 1e-5), torch.float16: (1e-2, 1e-3), torch.bfloat16: (2e-2, 1e-2)}

OPS = {
    "softmax": Op("softmax", _softmax, _softmax_inputs,
                  shapes=((1, 1), (3, 7), (64, 1000), (128, 4096), (5, 33)),
                  bench_shape=(8192, 4096), tol=TOL,
                  banned=("torch.softmax", "F.softmax", ".softmax(", "log_softmax", "kernelladder")),
    "rmsnorm": Op("rmsnorm", _rmsnorm, _rmsnorm_inputs,
                  shapes=((1, 1), (3, 7), (64, 1000), (128, 4096), (5, 33)),
                  bench_shape=(8192, 4096), tol=TOL,
                  banned=("rms_norm", "RMSNorm", "kernelladder")),
    "attention": Op("attention", _attention, _attention_inputs,
                    shapes=((1, 1, 1, 16), (2, 3, 17, 32), (1, 2, 128, 64), (2, 4, 257, 64)),
                    bench_shape=(4, 16, 1024, 64), tol={torch.float32: (1e-4, 1e-4), torch.float16: (2e-2, 2e-3), torch.bfloat16: (5e-2, 1e-2)},
                    traffic_bytes=lambda ins: 4 * ins[0].numel() * ins[0].element_size(),
                    banned=("scaled_dot_product_attention", "flash_attn", "torch.softmax", "F.softmax", ".softmax(",
                            "xformers", "kernelladder")),
    "int8_matmul": Op("int8_matmul", _int8_matmul, _int8_inputs,
                      shapes=((1, 1, 1), (5, 33, 7), (64, 128, 96), (130, 512, 70)),
                      bench_shape=(4096, 1024, 4096), tol=TOL,
                      traffic_bytes=lambda ins: ins[0].numel() + ins[1].numel() + ins[0].shape[0] * ins[1].shape[1] * ins[2].element_size(),
                      banned=("_int_mm", "_scaled_mm", "kernelladder")),
    "kv_quant": Op("kv_quant", _kv_quant, _kv_inputs,
                   shapes=((1, 1, 1, 8), (2, 3, 5, 64), (1, 8, 33, 128), (2, 4, 1, 16)),
                   bench_shape=(16, 32, 1024, 128), tol=TOL,
                   traffic_bytes=lambda ins: ins[0].numel() * (ins[0].element_size() + 1),
                   banned=("kernelladder",)),
}
