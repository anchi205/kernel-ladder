import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(x_ptr, w_ptr, out_ptr, rows, N, eps,
                    BLOCK: tl.constexpr, ROWS: tl.constexpr):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    # weight loaded once per program, kept in registers across rows
    w = tl.load(w_ptr + cols, mask=mask, other=0.0,
                eviction_policy="evict_last").to(tl.float32)
    inv_n = 1.0 / N
    base_row = pid * ROWS
    for r in tl.static_range(ROWS):
        row = base_row + r
        valid = row < rows
        m = mask & valid
        offs = row.to(tl.int64) * N + cols
        x = tl.load(x_ptr + offs, mask=m, other=0.0,
                    eviction_policy="evict_first").to(tl.float32)
        ms = tl.sum(x * x, axis=0) * inv_n
        inv = tl.rsqrt(ms + eps)
        y = x * inv * w
        tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty), mask=m)


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
    elif BLOCK <= 4096:
        num_warps = 4
    else:
        num_warps = 8
    ROWS = 4 if (rows >= 4096 and BLOCK <= 4096) else 1
    grid = (triton.cdiv(rows, ROWS),)
    _rmsnorm_kernel[grid](x2, w, out, rows, N, eps,
                          BLOCK=BLOCK, ROWS=ROWS, num_warps=num_warps)
    return out.view(orig_shape)
