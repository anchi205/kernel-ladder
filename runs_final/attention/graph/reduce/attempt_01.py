import torch
import triton
import triton.language as tl


@triton.jit
def _attn_fwd(
    Q, K, V, O,
    N, qk_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    base = pid_bh.to(tl.int64) * N * D
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    q_ptrs = Q + base + offs_m[:, None] * D + offs_d[None, :]
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N, other=0.0)

    m_i = tl.full([BLOCK_M], -1e30, dtype=tl.float32)
    # partial row sums kept per-element in registers; reduced once at the end
    l_t = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    diag_start = pid_m * BLOCK_M

    # Fully-visible blocks: no masks needed (all offs_n < diag_start < N)
    for start_n in range(0, diag_start, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k = tl.load(K + base + offs_n[:, None] * D + offs_d[None, :])
        v = tl.load(V + base + offs_n[:, None] * D + offs_d[None, :])

        s = tl.dot(q, tl.trans(k), input_precision="ieee") * qk_scale
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_t = l_t * alpha[:, None] + p
        acc = acc * alpha[:, None] + tl.dot(p, v, input_precision="ieee")
        m_i = m_new

    # Diagonal block(s): masked
    hi = tl.minimum(diag_start + BLOCK_M, N)
    for start_n in range(diag_start, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        kv_mask = offs_n[:, None] < N
        k = tl.load(K + base + offs_n[:, None] * D + offs_d[None, :], mask=kv_mask, other=0.0)
        v = tl.load(V + base + offs_n[:, None] * D + offs_d[None, :], mask=kv_mask, other=0.0)

        s = tl.dot(q, tl.trans(k), input_precision="ieee") * qk_scale
        keep = (offs_m[:, None] >= offs_n[None, :]) & (offs_n[None, :] < N)
        s = tl.where(keep, s, -1e30)

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        p = tl.where(keep, p, 0.0)
        l_t = l_t * alpha[:, None] + p
        acc = acc * alpha[:, None] + tl.dot(p, v, input_precision="ieee")
        m_i = m_new

    l_i = tl.sum(l_t, axis=1)
    acc = acc / l_i[:, None]
    o_ptrs = O + base + offs_m[:, None] * D + offs_d[None, :]
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=offs_m[:, None] < N)


def forward(q, k, v):
    B, H, N, D = q.shape
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    o = torch.empty_like(q)
    BLOCK_M = 64
    BLOCK_N = 64
    grid = (triton.cdiv(N, BLOCK_M), B * H)
    _attn_fwd[grid](
        q, k, v, o, N, (D ** -0.5) * 1.4426950408889634,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=D,
        num_warps=4, num_stages=2,
    )
    return o
