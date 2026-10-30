import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _kv_quant_kernel(x_ptr, q_ptr, s_ptr, n_rows, D,
                     ROWS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK)
    rmask = rows < n_rows
    cmask = cols < D
    mask = rmask[:, None] & cmask[None, :]
    base = rows.to(tl.int64)[:, None] * D + cols[None, :]
    x = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=1)
    amax = tl.maximum(amax, 1e-8)
    scale = libdevice.div_rn(amax, 127.0)
    v = libdevice.div_rn(x, scale[:, None])
    v = libdevice.rint(v)
    v = tl.minimum(tl.maximum(v, -127.0), 127.0)
    tl.store(q_ptr + base, v.to(tl.int8), mask=mask)
    tl.store(s_ptr + rows, scale, mask=rmask)


def forward(x):
    x = x.contiguous()
    D = x.shape[-1]
    rows = x.numel() // D
    q = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    scale = torch.empty(x.shape[:-1], dtype=torch.float32, device=x.device)
    BLOCK = triton.next_power_of_2(D)
    # target ~1024 elements per program
    ROWS = max(1, min(64, 1024 // BLOCK))
    grid = (triton.cdiv(rows, ROWS),)
    _kv_quant_kernel[grid](x, q, scale, rows, D, ROWS=ROWS, BLOCK=BLOCK, num_warps=4)
    return q, scale
