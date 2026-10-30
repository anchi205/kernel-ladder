import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(x_ptr, w_ptr, out_ptr, rows, N, eps,
                    BLOCK: tl.constexpr, R: tl.constexpr):
    pid = tl.program_id(0)
    r_idx = pid * R + tl.arange(0, R)            # [R]
    cols = tl.arange(0, BLOCK)                   # [BLOCK]
    cmask = cols < N
    rmask = r_idx < rows
    mask = rmask[:, None] & cmask[None, :]
    offs = r_idx.to(tl.int64)[:, None] * N + cols[None, :]
    w = tl.load(w_ptr + cols, mask=cmask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=1) * (1.0 / N)       # [R]
    inv = tl.rsqrt(ms + eps)
    y = x * inv[:, None] * w[None, :]
    tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


def forward(x, w, eps: float = 1e-6):
    orig_shape = x.shape
    N = orig_shape[-1]
    x2 = x.contiguous().view(-1, N)
    w = w.contiguous()
    out = torch.empty_like(x2)
    rows = x2.shape[0]
    BLOCK = triton.next_power_of_2(N)
    if BLOCK <= 256:
        num_warps = 2
        R = 1
    elif BLOCK <= 1024:
        num_warps = 4
        R = 1
    elif BLOCK <= 4096:
        num_warps = 8
        R = 2 if rows >= 1024 else 1
    else:
        num_warps = 16
        R = 1
    grid = (triton.cdiv(rows, R),)
    _rmsnorm_kernel[grid](x2, w, out, rows, N, eps,
                          BLOCK=BLOCK, R=R, num_warps=num_warps)
    return out.view(orig_shape)
