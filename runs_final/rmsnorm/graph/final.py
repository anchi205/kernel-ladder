import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(x_ptr, w_ptr, out_ptr, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + eps)
    y = x * inv * w
    tl.store(out_ptr + row * N + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


def forward(x, w, eps: float = 1e-6):
    orig_shape = x.shape
    N = orig_shape[-1]
    x2 = x.contiguous().view(-1, N)
    w = w.contiguous()
    out = torch.empty_like(x2)
    rows = x2.shape[0]
    BLOCK = triton.next_power_of_2(N)
    num_warps = 4 if BLOCK <= 2048 else 8
    _rmsnorm_kernel[(rows,)](x2, w, out, N, eps, BLOCK=BLOCK, num_warps=num_warps)
    return out.view(orig_shape)
