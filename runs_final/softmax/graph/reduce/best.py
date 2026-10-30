import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(x_ptr, out_ptr, n_cols, stride_x, stride_o,
                    BLOCK: tl.constexpr, EVEN: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    if EVEN:
        x = tl.load(x_ptr + row * stride_x + offs)
    else:
        mask = offs < n_cols
        x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=-float("inf"))
    m = tl.max(x, axis=0)
    e = tl.exp2((x - m) * 1.4426950408889634)
    s = tl.sum(e, axis=0)
    y = e * (1.0 / s)
    if EVEN:
        tl.store(out_ptr + row * stride_o + offs, y, cache_modifier=".cs")
    else:
        tl.store(out_ptr + row * stride_o + offs, y, mask=mask, cache_modifier=".cs")


def forward(x):
    orig_shape = x.shape
    x2 = x.reshape(-1, orig_shape[-1]).contiguous()
    n_rows, n_cols = x2.shape
    out = torch.empty_like(x2)
    BLOCK = triton.next_power_of_2(n_cols)
    if BLOCK <= 128:
        num_warps = 1
    elif BLOCK <= 512:
        num_warps = 2
    elif BLOCK <= 2048:
        num_warps = 4
    else:
        num_warps = 8
    _softmax_kernel[(n_rows,)](
        x2, out, n_cols, x2.stride(0), out.stride(0),
        BLOCK=BLOCK, EVEN=(BLOCK == n_cols), num_warps=num_warps,
    )
    return out.reshape(orig_shape)
