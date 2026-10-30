# Known-good: tiled int8 matmul with int32 accumulation and a fused scale epilogue.
import torch
import triton
import triton.language as tl


@triton.jit
def _kernel(a_ptr, b_ptr, sa_ptr, sb_ptr, out_ptr, M, N, K,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm, pn = tl.program_id(0), tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.int32)
    for k in range(0, K, BK):
        a = tl.load(a_ptr + rm[:, None] * K + (k + rk)[None, :],
                    mask=(rm[:, None] < M) & ((k + rk)[None, :] < K), other=0)
        b = tl.load(b_ptr + (k + rk)[:, None] * N + rn[None, :],
                    mask=((k + rk)[:, None] < K) & (rn[None, :] < N), other=0)
        acc = tl.dot(a, b, acc)
    sa = tl.load(sa_ptr + rm, mask=rm < M, other=0.0).to(tl.float32)
    sb = tl.load(sb_ptr + rn, mask=rn < N, other=0.0).to(tl.float32)
    res = acc.to(tl.float32) * sa[:, None] * sb[None, :]
    tl.store(out_ptr + rm[:, None] * N + rn[None, :], res.to(out_ptr.dtype.element_ty),
             mask=(rm[:, None] < M) & (rn[None, :] < N))


def forward(a, b, sa, sb):
    M, K = a.shape
    N = b.shape[1]
    out = torch.empty((M, N), dtype=sa.dtype, device=a.device)
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    _kernel[grid](a.contiguous(), b.contiguous(), sa, sb, out, M, N, K, BM=64, BN=64, BK=32)
    return out
