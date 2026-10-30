import torch
import triton
import triton.language as tl

_SM_COUNT = {}


@triton.jit
def _softmax_kernel(x_ptr, out_ptr, n_rows, n_cols, stride_x, stride_o,
                    BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    offs = tl.arange(0, BLOCK)
    cmask = offs < n_cols

    row = pid.to(tl.int64)
    x = tl.load(x_ptr + row * stride_x + offs, mask=cmask & (row < n_rows),
                other=-float("inf"))
    iters = tl.cdiv(n_rows - pid, nprog)
    for _ in range(0, iters):
        nrow = row + nprog
        xn = tl.load(x_ptr + nrow * stride_x + offs,
                     mask=cmask & (nrow < n_rows), other=-float("inf"))
        m = tl.max(x, axis=0)
        e = tl.exp2((x - m) * 1.4426950408889634)
        s = tl.sum(e, axis=0)
        y = e * (1.0 / s)
        tl.store(out_ptr + row * stride_o + offs, y, mask=cmask & (row < n_rows),
                 cache_modifier=".cs")
        x = xn
        row = nrow


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
    elif BLOCK <= 4096:
        num_warps = 4
    elif BLOCK <= 16384:
        num_warps = 8
    else:
        num_warps = 16
    dev = x2.device
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    if idx not in _SM_COUNT:
        _SM_COUNT[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    grid = max(1, min(n_rows, _SM_COUNT[idx] * 8))
    _softmax_kernel[(grid,)](
        x2, out, n_rows, n_cols, x2.stride(0), out.stride(0),
        BLOCK=BLOCK, num_warps=num_warps,
    )
    return out.reshape(orig_shape)
