import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_single(x_ptr, w_ptr, out_ptr, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_row = x_ptr + row.to(tl.int64) * N
    o_row = out_ptr + row.to(tl.int64) * N
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(mean + eps)
    y = x * rstd * w
    tl.store(o_row + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _rmsnorm_loop(x_ptr, w_ptr, out_ptr, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_row = x_ptr + row.to(tl.int64) * N
    o_row = out_ptr + row.to(tl.int64) * N

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        acc += x * x
    mean = tl.sum(acc, axis=0) / N
    rstd = 1.0 / tl.sqrt(mean + eps)

    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = x * rstd * w
        tl.store(o_row + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


def forward(x, w, eps: float = 1e-6):
    orig_shape = x.shape
    N = orig_shape[-1]
    x2 = x.contiguous().view(-1, N)
    w = w.contiguous()
    out = torch.empty_like(x2)
    rows = x2.shape[0]
    P = triton.next_power_of_2(N)
    if P <= 16384:
        if P >= 8192:
            nw = 16
        elif P >= 4096:
            nw = 8
        elif P >= 1024:
            nw = 4
        elif P >= 256:
            nw = 2
        else:
            nw = 1
        _rmsnorm_single[(rows,)](x2, w, out, N, eps, BLOCK=P, num_warps=nw)
    else:
        _rmsnorm_loop[(rows,)](x2, w, out, N, eps, BLOCK=2048, num_warps=8)
    return out.view(orig_shape)
