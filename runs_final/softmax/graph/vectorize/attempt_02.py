import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(x_ptr, out_ptr, n_rows, n_cols, stride_x, stride_o,
                    BLOCK: tl.constexpr, EVEN_N: tl.constexpr,
                    ROWS_PER_PROG: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    offs = tl.arange(0, BLOCK)
    LOG2E: tl.constexpr = 1.4426950408889634
    if not EVEN_N:
        mask = offs < n_cols
    for i in range(ROWS_PER_PROG):
        r = (pid + i * nprog).to(tl.int64)
        if r < n_rows:
            x_row = x_ptr + r * stride_x
            o_row = out_ptr + r * stride_o
            if EVEN_N:
                x = tl.load(x_row + offs)
            else:
                x = tl.load(x_row + offs, mask=mask, other=-float("inf"))
            m = tl.max(x, axis=0)
            e = tl.exp2((x - m) * LOG2E)
            if not EVEN_N:
                e = tl.where(mask, e, 0.0)
            s = tl.sum(e, axis=0)
            y = e * (1.0 / s)
            if EVEN_N:
                tl.store(o_row + offs, y)
            else:
                tl.store(o_row + offs, y, mask=mask)


def forward(x):
    orig_shape = x.shape
    x2 = x.reshape(-1, orig_shape[-1]).contiguous()
    n_rows, n_cols = x2.shape
    out = torch.empty_like(x2)
    BLOCK = triton.next_power_of_2(n_cols)
    if BLOCK >= 4096:
        num_warps = 8
    elif BLOCK >= 1024:
        num_warps = 4
    elif BLOCK >= 256:
        num_warps = 2
    else:
        num_warps = 1
    # a few rows per program, grid stays large enough for good balance
    rows_per_prog = 4 if n_rows >= 4096 else 1
    grid = (triton.cdiv(n_rows, rows_per_prog),)
    _softmax_kernel[grid](
        x2, out, n_rows, n_cols, x2.stride(0), out.stride(0),
        BLOCK=BLOCK, EVEN_N=(BLOCK == n_cols), ROWS_PER_PROG=rows_per_prog,
        num_warps=num_warps,
    )
    return out.reshape(orig_shape)
