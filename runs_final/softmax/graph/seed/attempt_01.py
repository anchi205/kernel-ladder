import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(x_ptr, out_ptr, n_cols, stride_x, stride_o, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=-float("inf"))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    tl.store(out_ptr + row * stride_o + offs, e / s, mask=mask)


def forward(x):
    orig_shape = x.shape
    x2 = x.reshape(-1, orig_shape[-1]).contiguous()
    n_rows, n_cols = x2.shape
    out = torch.empty_like(x2)
    BLOCK = triton.next_power_of_2(n_cols)
    num_warps = 4 if BLOCK <= 2048 else 8
    _softmax_kernel[(n_rows,)](
        x2, out, n_cols, x2.stride(0), out.stride(0),
        BLOCK=BLOCK, num_warps=num_warps,
    )
    return out.reshape(orig_shape)
