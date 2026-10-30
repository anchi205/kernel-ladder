import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(x_ptr, w_ptr, out_ptr, rows, N, eps, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    r0 = pid * 2
    r1 = r0 + 1
    cols = tl.arange(0, BLOCK)
    cmask = cols < N
    m0 = cmask & (r0 < rows)
    m1 = cmask & (r1 < rows)
    # issue both row loads before any compute for more bytes in flight
    x0 = tl.load(x_ptr + r0 * N + cols, mask=m0, other=0.0).to(tl.float32)
    x1 = tl.load(x_ptr + r1 * N + cols, mask=m1, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=cmask, other=0.0).to(tl.float32)
    inv0 = tl.math.rsqrt(tl.sum(x0 * x0, axis=0) / N + eps)
    inv1 = tl.math.rsqrt(tl.sum(x1 * x1, axis=0) / N + eps)
    y0 = x0 * inv0 * w
    y1 = x1 * inv1 * w
    tl.store(out_ptr + r0 * N + cols, y0.to(out_ptr.dtype.element_ty), mask=m0)
    tl.store(out_ptr + r1 * N + cols, y1.to(out_ptr.dtype.element_ty), mask=m1)


def forward(x, w, eps: float = 1e-6):
    orig_shape = x.shape
    N = orig_shape[-1]
    x2 = x.contiguous().view(-1, N)
    w = w.contiguous()
    out = torch.empty_like(x2)
    rows = x2.shape[0]
    BLOCK = triton.next_power_of_2(N)
    if BLOCK <= 1024:
        num_warps = 4
    elif BLOCK <= 4096:
        num_warps = 8
    else:
        num_warps = 8
    grid = ((rows + 1) // 2,)
    _rmsnorm_kernel[grid](x2, w, out, rows, N, eps, BLOCK=BLOCK, num_warps=num_warps)
    return out.view(orig_shape)
