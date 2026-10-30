import torch
import triton
import triton.language as tl


_configs = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
]


@triton.autotune(configs=_configs, key=['M', 'N', 'K'])
@triton.heuristics({'EVEN_K': lambda args: args['K'] % args['BLOCK_K'] == 0})
@triton.jit
def _int8_mm_kernel(
    a_ptr, b_ptr, sa_ptr, sb_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_in_group = GROUP_M * num_pid_n
    group_id = pid // num_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # clamp rows/cols so loads need no M/N masks (results for padded rows are discarded)
    offs_m_c = tl.where(offs_m < M, offs_m, 0)
    offs_n_c = tl.where(offs_n < N, offs_n, 0)

    a_ptrs = a_ptr + offs_m_c[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n_c[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for k0 in range(0, K, BLOCK_K):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            k_rem = K - k0
            a = tl.load(a_ptrs, mask=offs_k[None, :] < k_rem, other=0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < k_rem, other=0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

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

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    _int8_mm_kernel[grid](
        a, b, sa, sb, out,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
    )
    return out
