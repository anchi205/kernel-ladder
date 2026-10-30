# Hand-written reference solution: one fused Triton kernel, one program per row.
import torch
import triton
import triton.language as tl


@triton.jit
def _rms(x_ptr, w_ptr, out_ptr, n, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n
    x = tl.load(x_ptr + row * n + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.rsqrt(tl.sum(x * x, axis=0) / n + eps)
    tl.store(out_ptr + row * n + cols, (x * r * w).to(out_ptr.dtype.element_ty), mask=mask)


def forward(x, w, eps: float = 1e-6):
    x = x.contiguous()
    out = torch.empty_like(x)
    rows, n = x.shape
    _rms[(rows,)](x, w, out, n, eps, BLOCK=triton.next_power_of_2(n))
    return out
