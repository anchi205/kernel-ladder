import torch
import triton
import triton.language as tl


def _cfg(bm, bn, bk, w, s, g=8):
    return triton.Config({'BLOCK_M': bm, 'BLOCK_N': bn, 'BLOCK_K': bk, 'GROUP_M': g},
                         num_warps=w, num_stages=s)


_configs = [
    _cfg(128, 128, 64, 8, 3),
    _cfg(128, 128, 64, 4, 4),
    _cfg(128, 128, 64, 4, 3),
    _cfg(128, 128, 128, 8, 2),
    _cfg(128, 128, 128, 4, 2),
    _cfg(128, 256, 64, 8, 3),
    _cfg(256, 128, 64, 8, 3),
    _cfg(128, 256, 32, 8, 4),
    _cfg(256, 128, 32, 8, 4),
    _cfg(128, 64, 128, 4, 3),
    _cfg(64, 128, 128, 4, 3),
    _cfg(128, 128, 32, 4, 4),
    _cfg(128, 128, 32, 4, 5),
    _cfg(64, 64, 128, 4, 3),
    _cfg(64, 256, 64, 8, 3),
    _cfg(256, 64, 64, 8, 3),
    _cfg(128, 128, 64, 8, 4, 16),
    _cfg(128, 256, 64, 8, 3, 4),
]


@triton.autotune(configs=_configs, key=['M', 'N', 'K'])
@triton.heuristics({'EVEN_K': lambda args: args['K'] % args['BLOCK_K'] == 0})
@triton.jit
def _int8_mm_kernel(
    a_ptr, bt_ptr, sa_ptr, sb_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
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

    offs_m_c = tl.where(offs_m < M, offs_m, 0)
    offs_n_c = tl.where(offs_n < N, offs_n, 0)

    a_ptrs = a_ptr + offs_m_c[:, None] * stride_am + offs_k[None, :] * stride_ak
    # B is pre-transposed to (N, K): K-contiguous tile of shape (BLOCK_N, BLOCK_K)
    b_ptrs = bt_ptr + offs_n_c[:, None] * stride_bn + offs_k[None, :] * stride_bk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for k0 in range(0, K, BLOCK_K):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            k_rem = K - k0
            a = tl.load(a_ptrs, mask=offs_k[None, :] < k_rem, other=0)
            b = tl.load(b_ptrs, mask=offs_k[None, :] < k_rem, other=0)
        acc = tl.dot(a, tl.trans(b), acc)
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
    a = a.contiguous()
    bt = b.t().contiguous()  # (N, K), K-contiguous
    sa = sa.contiguous()
    sb = sb.contiguous()
    out = torch.empty((M, N), dtype=sa.dtype, device=a.device)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    _int8_mm_kernel[grid](
        a, bt, sa, sb, out,
        M, N, K,
        a.stride(0), a.stride(1),
        bt.stride(0), bt.stride(1),
        out.stride(0), out.stride(1),
    )
    return out
