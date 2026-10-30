# Known-good: one fused kernel, one program per (batch, head, token) row.
import torch
import triton
import triton.language as tl


@triton.jit
def _kernel(x_ptr, q_ptr, s_ptr, d, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    m = cols < d
    x = tl.load(x_ptr + row * d + cols, mask=m, other=0.0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-8) / 127
    v = tl.floor(x / scale + 0.5)
    v = tl.minimum(tl.maximum(v, -127.0), 127.0)
    tl.store(q_ptr + row * d + cols, v.to(tl.int8), mask=m)
    tl.store(s_ptr + row, scale)


def forward(x):
    x = x.contiguous()
    q = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    s = torch.empty(x.shape[:-1], dtype=torch.float32, device=x.device)
    d = x.shape[-1]
    _kernel[(x.numel() // d,)](x, q, s, d, BLOCK=triton.next_power_of_2(d))
    return q, s
