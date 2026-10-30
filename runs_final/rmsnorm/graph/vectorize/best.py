import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(x_ptr, w_ptr, out_ptr, N, eps, BLOCK: tl.constexpr, ALIGNED: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    if ALIGNED:
        cols = tl.max_contiguous(tl.multiple_of(cols, 4), BLOCK)
    mask = cols < N
    base = row * N
    if ALIGNED:
        base = tl.multiple_of(base, 4)
    x = tl.load(x_ptr + base + cols, mask=mask, other=0.0,
                eviction_policy="evict_first").to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0,
                eviction_policy="evict_last").to(tl.float32)
    ms = tl.sum(x * x, axis=0) / N
    inv = tl.rsqrt(ms + eps)
    y = x * inv * w
    tl.store(out_ptr + base + cols, y.to(out_ptr.dtype.element_ty), mask=mask,
             eviction_policy="evict_first")


def forward(x, w, eps: float = 1e-6):
    orig_shape = x.shape
    N = orig_shape[-1]
    x2 = x.contiguous().view(-1, N)
    w = w.contiguous()
    out = torch.empty_like(x2)
    rows = x2.shape[0]
    BLOCK = triton.next_power_of_2(N)
    if BLOCK <= 1024:
        num_warps = 4 if BLOCK >= 512 else 1 if BLOCK <= 64 else 2
    elif BLOCK <= 2048:
        num_warps = 4
    else:
        num_warps = 8
    aligned = (N % 4 == 0)
    _rmsnorm_kernel[(rows,)](x2, w, out, N, eps, BLOCK=BLOCK, ALIGNED=aligned,
                             num_warps=num_warps)
    return out.view(orig_shape)
