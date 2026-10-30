import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _kv_quant_kernel(x_ptr, q_ptr, s_ptr, D, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    x = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=0)
    amax = tl.maximum(amax, 1e-8)
    scale = libdevice.div_rn(amax, 127.0)
    y = libdevice.div_rn(x, scale)
    y = libdevice.rint(y)
    y = tl.minimum(tl.maximum(y, -127.0), 127.0)
    tl.store(q_ptr + row * D + offs, y.to(tl.int8), mask=mask)
    tl.store(s_ptr + row, scale)


def forward(x):
    x = x.contiguous()
    D = x.shape[-1]
    rows = x.numel() // D
    q = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    scale = torch.empty(x.shape[:-1], dtype=torch.float32, device=x.device)
    BLOCK = triton.next_power_of_2(D)
    nw = 1 if BLOCK <= 128 else 4
    _kv_quant_kernel[(rows,)](x, q, scale, D, BLOCK=BLOCK, num_warps=nw)
    return q, scale
