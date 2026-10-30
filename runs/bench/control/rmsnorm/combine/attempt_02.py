import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_row_kernel(x_ptr, w_ptr, out_ptr, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0,
                eviction_policy="evict_first").to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0,
                eviction_policy="evict_last").to(tl.float32)
    ss = tl.sum(x * x, axis=0)
    rstd = tl.rsqrt(ss / N + eps)
    y = x * rstd * w
    tl.store(out_ptr + row * N + offs, y.to(out_ptr.dtype.element_ty), mask=mask,
             eviction_policy="evict_first")


@triton.jit
def _rmsnorm_loop_kernel(x_ptr, w_ptr, out_ptr, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    x_row = x_ptr + row * N
    o_row = out_ptr + row * N

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        acc += x * x
    rstd = tl.rsqrt(tl.sum(acc, axis=0) / N + eps)

    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = x * rstd * w
        tl.store(o_row + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


def forward(x, w, eps: float = 1e-6):
    N = x.shape[-1]
    if not x.is_contiguous():
        x = x.contiguous()
    if not w.is_contiguous():
        w = w.contiguous()
    out = torch.empty_like(x)
    rows = x.numel() // N
    if rows == 0:
        return out
    P = 1 << (N - 1).bit_length() if N > 1 else 1
    if P <= 8192:
        if P >= 4096:
            nw = 8
        elif P >= 1024:
            nw = 4
        elif P >= 256:
            nw = 2
        else:
            nw = 1
        _rmsnorm_row_kernel[(rows,)](x, w, out, N, eps, BLOCK=P, num_warps=nw)
    else:
        _rmsnorm_loop_kernel[(rows,)](x, w, out, N, eps, BLOCK=2048, num_warps=8)
    return out
