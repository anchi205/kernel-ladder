import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(x_ptr, out_ptr, n_cols, stride_x, stride_o,
                    BLOCK: tl.constexpr, EVEN_N: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    offs = tl.max_contiguous(tl.multiple_of(offs, 4), 4)
    x_row = x_ptr + row * stride_x
    o_row = out_ptr + row * stride_o
    if EVEN_N:
        x = tl.load(x_row + offs, eviction_policy="evict_first")
    else:
        mask = offs < n_cols
        x = tl.load(x_row + offs, mask=mask, other=-float("inf"),
                    eviction_policy="evict_first")
    m = tl.max(x, axis=0)
    LOG2E: tl.constexpr = 1.4426950408889634
    e = tl.exp2((x - m) * LOG2E)
    if not EVEN_N:
        e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    r = 1.0 / s
    y = e * r
    if EVEN_N:
        tl.store(o_row + offs, y, cache_modifier=".cs")
    else:
        tl.store(o_row + offs, y, mask=mask, cache_modifier=".cs")


def forward(x):
    orig_shape = x.shape
    x2 = x.reshape(-1, orig_shape[-1]).contiguous()
    n_rows, n_cols = x2.shape
    out = torch.empty_like(x2)
    BLOCK = triton.next_power_of_2(n_cols)
    if BLOCK <= 4096:
        num_warps = 4 if BLOCK >= 1024 else (2 if BLOCK >= 256 else 1)
    else:
        num_warps = 8
    _softmax_kernel[(n_rows,)](
        x2, out, n_cols, x2.stride(0), out.stride(0),
        BLOCK=BLOCK, EVEN_N=(BLOCK == n_cols), num_warps=num_warps,
    )
    return out.reshape(orig_shape)
