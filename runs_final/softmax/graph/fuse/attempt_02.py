import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(x_ptr, out_ptr, n_rows, n_cols, stride_x, stride_o,
                    BLOCK: tl.constexpr, NUM_STAGES: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    for row in tl.range(pid, n_rows, nprog, num_stages=NUM_STAGES):
        x = tl.load(x_ptr + row.to(tl.int64) * stride_x + offs, mask=mask,
                    other=-float("inf"))
        m = tl.max(x, axis=0)
        e = tl.exp2((x - m) * 1.4426950408889634)
        s = tl.sum(e, axis=0)
        r = 1.0 / s
        tl.store(out_ptr + row.to(tl.int64) * stride_o + offs, e * r, mask=mask)


_SM_COUNT = {}


def forward(x):
    orig_shape = x.shape
    x2 = x.reshape(-1, orig_shape[-1]).contiguous()
    n_rows, n_cols = x2.shape
    out = torch.empty_like(x2)
    BLOCK = triton.next_power_of_2(n_cols)
    if BLOCK <= 1024:
        num_warps = 4
    elif BLOCK <= 4096:
        num_warps = 8
    else:
        num_warps = 8
    dev = x2.device
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    if idx not in _SM_COUNT:
        _SM_COUNT[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    sms = _SM_COUNT[idx]
    grid = (max(1, min(n_rows, sms * 4)),)
    _softmax_kernel[grid](
        x2, out, n_rows, n_cols, x2.stride(0), out.stride(0),
        BLOCK=BLOCK, NUM_STAGES=2, num_warps=num_warps,
    )
    return out.reshape(orig_shape)
