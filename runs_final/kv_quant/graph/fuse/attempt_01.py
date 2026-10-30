import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _kv_quant_kernel(x_ptr, q_ptr, s_ptr, n_rows, D,
                     BLOCK: tl.constexpr, ROWS: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    r = pid * ROWS + tl.arange(0, ROWS).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    rmask = r < n_rows
    mask = rmask[:, None] & (offs[None, :] < D)
    ptrs = r[:, None] * D + offs[None, :]
    x = tl.load(x_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=1)
    amax = tl.maximum(amax, 1e-8)
    scale = libdevice.div_rn(amax, 127.0)
    y = libdevice.div_rn(x, scale[:, None])
    y = libdevice.rint(y)
    y = tl.minimum(tl.maximum(y, -127.0), 127.0)
    tl.store(q_ptr + ptrs, y.to(tl.int8), mask=mask)
    tl.store(s_ptr + r, scale, mask=rmask)


def forward(x):
    x = x.contiguous()
    D = x.shape[-1]
    rows = x.numel() // D
    q = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    scale = torch.empty(x.shape[:-1], dtype=torch.float32, device=x.device)
    BLOCK = triton.next_power_of_2(D)
    ROWS = max(1, min(8, 1024 // BLOCK))
    nw = 4 if ROWS * BLOCK >= 512 else 1
    grid = (triton.cdiv(rows, ROWS),)
    _kv_quant_kernel[grid](x, q, scale, rows, D, BLOCK=BLOCK, ROWS=ROWS, num_warps=nw)
    return q, scale
