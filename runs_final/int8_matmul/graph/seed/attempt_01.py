import torch
import triton
import triton.language as tl


@triton.jit
def _int8_mm_kernel(
    a_ptr, b_ptr, sa_ptr, sb_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for k0 in range(0, K, BLOCK_K):
        k_idx = k0 + offs_k
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + k_idx[None, :] * stride_ak
        b_ptrs = b_ptr + k_idx[:, None] * stride_bk + offs_n[None, :] * stride_bn
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k_idx[None, :] < K), other=0)
        b = tl.load(b_ptrs, mask=(k_idx[:, None] < K) & (offs_n[None, :] < N), other=0)
        acc = tl.dot(a, b, acc)

    sa = tl.load(sa_ptr + offs_m, mask=offs_m < M, other=0.0).to(tl.float32)
    sb = tl.load(sb_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    out = acc.to(tl.float32) * sa[:, None] * sb[None, :]

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, out.to(c_ptr.dtype.element_ty), mask=c_mask)


def forward(a, b, sa, sb):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    out = torch.empty((M, N), dtype=sa.dtype, device=a.device)

    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _int8_mm_kernel[grid](
        a, b, sa, sb, out,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4,
    )
    return out
