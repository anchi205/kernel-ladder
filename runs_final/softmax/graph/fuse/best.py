import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(x_ptr, out_ptr, n_cols, stride_x, stride_o, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=-float("inf"),
                cache_modifier=".cg")
    m = tl.max(x, axis=0)
    # exp(x - m) == exp2((x - m) * log2(e)); masked lanes give exp2(-inf) = 0
    e = tl.exp2((x - m) * 1.4426950408889634)
    s = tl.sum(e, axis=0)
    r = 1.0 / s
    tl.store(out_ptr + row * stride_o + offs, e * r, mask=mask,
             eviction_policy="evict_first")


def forward(x):
    orig_shape = x.shape
    x2 = x.reshape(-1, orig_shape[-1]).contiguous()
    n_rows, n_cols = x2.shape
    out = torch.empty_like(x2)
    BLOCK = triton.next_power_of_2(n_cols)
    if BLOCK <= 1024:
        num_warps = 4
    elif BLOCK <= 4096:
        num_warps = 4
    else:
        num_warps = 8
    _softmax_kernel[(n_rows,)](
        x2, out, n_cols, x2.stride(0), out.stride(0),
        BLOCK=BLOCK, num_warps=num_warps,
    )
    return out.reshape(orig_shape)
