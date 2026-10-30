import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _kv_quant_kernel(x_ptr, q_ptr, s_ptr, rows, D,
                     ROWS: tl.constexpr, BLOCK: tl.constexpr,
                     I64: tl.constexpr):
    pid = tl.program_id(0)
    if I64:
        r = pid.to(tl.int64) * ROWS + tl.arange(0, ROWS).to(tl.int64)
    else:
        r = pid * ROWS + tl.arange(0, ROWS)
    offs = tl.arange(0, BLOCK)
    rmask = r < rows
    cmask = offs < D
    mask = rmask[:, None] & cmask[None, :]
    ptrs = r[:, None] * D + offs[None, :]
    x = tl.load(x_ptr + ptrs, mask=mask, other=0.0,
                eviction_policy="evict_first").to(tl.float32)
    amax = tl.max(tl.abs(x), axis=1)
    amax = tl.maximum(amax, 1e-8)
    scale = libdevice.div_rn(amax, 127.0)
    v = libdevice.div_rn(x, scale[:, None])
    v = libdevice.rint(v)
    v = tl.minimum(tl.maximum(v, -127.0), 127.0)
    tl.store(q_ptr + ptrs, v.to(tl.int8), mask=mask)
    tl.store(s_ptr + r, scale, mask=rmask)


def forward(x):
    x = x.contiguous()
    D = x.shape[-1]
    rows = x.numel() // D
    q = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    scale = torch.empty(x.shape[:-1], dtype=torch.float32, device=x.device)
    BLOCK = triton.next_power_of_2(D)
    ROWS = max(1, min(64, 4096 // BLOCK))
    grid = (triton.cdiv(rows, ROWS),)
    i64 = x.numel() + ROWS * BLOCK >= 2**31
    nw = 8 if ROWS * BLOCK >= 4096 else 4
    _kv_quant_kernel[grid](x, q, scale, rows, D, ROWS=ROWS, BLOCK=BLOCK,
                           I64=i64, num_warps=nw)
    return q, scale
