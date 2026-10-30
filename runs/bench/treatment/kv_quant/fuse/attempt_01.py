import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _kv_quant_kernel(x_ptr, q_ptr, s_ptr, n_rows, D,
                     ROWS: tl.constexpr, BLOCK_D: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK_D)
    rmask = rows < n_rows
    mask = rmask[:, None] & (cols[None, :] < D)
    offs = rows[:, None].to(tl.int64) * D + cols[None, :]

    x = tl.load(x_ptr + offs, mask=mask, other=0.0,
                eviction_policy="evict_first").to(tl.float32)
    amax = tl.max(tl.abs(x), axis=1)
    amax = tl.maximum(amax, 1e-8)
    scale = tl.math.div_rn(amax, 127.0)

    y = tl.math.div_rn(x, scale[:, None])
    # round half to even (matches torch.round) via a single rint
    r = libdevice.rint(y)
    r = tl.minimum(tl.maximum(r, -127.0), 127.0)

    tl.store(q_ptr + offs, r.to(tl.int8), mask=mask)
    tl.store(s_ptr + rows, scale, mask=rmask)


def forward(x):
    x = x.contiguous()
    if x.dtype != torch.float32:
        x = x.float()
    D = x.shape[-1]
    n_rows = x.numel() // D
    q = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    s = torch.empty(x.shape[:-1], dtype=torch.float32, device=x.device)
    BLOCK_D = triton.next_power_of_2(D)
    ROWS = 8
    grid = (triton.cdiv(n_rows, ROWS),)
    _kv_quant_kernel[grid](x, q, s, n_rows, D, ROWS=ROWS, BLOCK_D=BLOCK_D,
                           num_warps=4)
    return q, s
