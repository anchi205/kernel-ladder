import torch
import triton
import triton.language as tl


@triton.jit
def _attn_fwd(
    Q, K, V, O,
    N, sm_scale,
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
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    hi = tl.minimum((pid_m + 1) * BLOCK_M, N)
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        kv_mask = offs_n[:, None] < N
        k = tl.load(K + base + offs_n[:, None] * D + offs_d[None, :], mask=kv_mask, other=0.0)
        v = tl.load(V + base + offs_n[:, None] * D + offs_d[None, :], mask=kv_mask, other=0.0)

        s = tl.dot(q, tl.trans(k), input_precision="ieee") * sm_scale
        keep = (offs_m[:, None] >= offs_n[None, :]) & (offs_n[None, :] < N)
        s = tl.where(keep, s, -1e30)

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        p = tl.where(keep, p, 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p, v, input_precision="ieee")
        m_i = m_new

    acc = acc / l_i[:, None]
    o_ptrs = O + base + offs_m[:, None] * D + offs_d[None, :]
    tl.store(o_ptrs, acc, mask=offs_m[:, None] < N)


def forward(q, k, v):
    B, H, N, D = q.shape
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    o = torch.empty_like(q)
    BLOCK_M = 32
    BLOCK_N = 32
    grid = (triton.cdiv(N, BLOCK_M), B * H)
    _attn_fwd[grid](
        q, k, v, o, N, D ** -0.5,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=D,
    )
    return o
