import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(x_ptr, w_ptr, out_ptr, R, N, eps,
                    BLOCK: tl.constexpr, ROWS: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    rows = pid * ROWS + tl.arange(0, ROWS).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    cmask = cols < N
    rmask = rows < R
    mask = rmask[:, None] & cmask[None, :]
    offs = rows[:, None] * N + cols[None, :]
    x = tl.load(x_ptr + offs, mask=mask, other=0.0,
                eviction_policy="evict_first").to(tl.float32)
    w = tl.load(w_ptr + cols, mask=cmask, other=0.0,
                eviction_policy="evict_last").to(tl.float32)
    ms = tl.sum(x * x, axis=1) / N
    inv = tl.rsqrt(ms + eps)
    y = x * inv[:, None] * w[None, :]
    tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty), mask=mask,
             eviction_policy="evict_first")


def forward(x, w, eps: float = 1e-6):
    orig_shape = x.shape
    N = orig_shape[-1]
    x2 = x.contiguous().view(-1, N)
    w = w.contiguous()
    out = torch.empty_like(x2)
    R = x2.shape[0]
    BLOCK = triton.next_power_of_2(N)
    if BLOCK >= 2048:
        ROWS = 2
        num_warps = 8
    elif BLOCK >= 512:
        ROWS = 1
        num_warps = 4
    else:
        ROWS = 1
        num_warps = 1 if BLOCK <= 64 else 2
    grid = (triton.cdiv(R, ROWS),)
    _rmsnorm_kernel[grid](x2, w, out, R, N, eps, BLOCK=BLOCK, ROWS=ROWS,
                          num_warps=num_warps)
    return out.view(orig_shape)
