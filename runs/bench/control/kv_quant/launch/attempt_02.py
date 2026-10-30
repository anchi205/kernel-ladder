import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _kv_quant_kernel(x_ptr, q_ptr, s_ptr, D, n_rows,
                     ROWS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    r = pid.to(tl.int64) * ROWS + tl.arange(0, ROWS).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    rmask = r < n_rows
    cmask = offs < D
    mask = rmask[:, None] & cmask[None, :]
    ptrs = r[:, None] * D + offs[None, :]
    x = tl.load(x_ptr + ptrs, mask=mask, other=0.0,
                cache_modifier=".cg",
                eviction_policy="evict_first").to(tl.float32)
    amax = tl.max(tl.abs(x), axis=1)
    amax = tl.maximum(amax, 1e-8)
    scale = libdevice.div_rn(amax, 127.0)
    v = libdevice.div_rn(x, scale[:, None])
    v = libdevice.rint(v)
    v = tl.minimum(tl.maximum(v, -127.0), 127.0)
    tl.store(q_ptr + ptrs, v.to(tl.int8), mask=mask,
             eviction_policy="evict_first")
    tl.store(s_ptr + r, scale, mask=rmask)


def forward(x):
    x = x.contiguous()
    D = x.shape[-1]
    rows = x.numel() // D
    q = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    scale = torch.empty(x.shape[:-1], dtype=torch.float32, device=x.device)
    BLOCK = triton.next_power_of_2(D)
    ROWS = max(1, 2048 // BLOCK)
    grid = (triton.cdiv(rows, ROWS),)
    _kv_quant_kernel[grid](x, q, scale, D, rows, ROWS=ROWS, BLOCK=BLOCK,
                           num_warps=8)
    return q, scale
