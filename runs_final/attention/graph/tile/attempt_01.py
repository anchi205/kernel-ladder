import torch
import triton
import triton.language as tl


@triton.jit
def _step(q, acc, m_i, l_i, K, V, base, offs_m, offs_d, start_n, N, qk_scale,
          BLOCK_N: tl.constexpr, D: tl.constexpr, MASK: tl.constexpr):
    offs_n = start_n + tl.arange(0, BLOCK_N)
    if MASK:
        kv_mask = offs_n[:, None] < N
        k = tl.load(K + base + offs_n[:, None] * D + offs_d[None, :], mask=kv_mask, other=0.0)
        v = tl.load(V + base + offs_n[:, None] * D + offs_d[None, :], mask=kv_mask, other=0.0)
    else:
        k = tl.load(K + base + offs_n[:, None] * D + offs_d[None, :])
        v = tl.load(V + base + offs_n[:, None] * D + offs_d[None, :])

    s = tl.dot(q, tl.trans(k), input_precision="ieee") * qk_scale
    if MASK:
        keep = (offs_m[:, None] >= offs_n[None, :]) & (offs_n[None, :] < N)
        s = tl.where(keep, s, -1e30)

    m_new = tl.maximum(m_i, tl.max(s, axis=1))
    alpha = tl.exp2(m_i - m_new)
    p = tl.exp2(s - m_new[:, None])
    if MASK:
        p = tl.where(keep, p, 0.0)
    l_i = l_i * alpha + tl.sum(p, axis=1)
    acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision="ieee")
    return acc, m_new, l_i


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
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    # Fully-visible blocks (no mask needed): keys [0, pid_m*BLOCK_M)
    full_end = (pid_m * BLOCK_M // BLOCK_N) * BLOCK_N
    full_end = tl.minimum(full_end, (N // BLOCK_N) * BLOCK_N)
    for start_n in range(0, full_end, BLOCK_N):
        acc, m_i, l_i = _step(q, acc, m_i, l_i, K, V, base, offs_m, offs_d,
                              start_n, N, qk_scale, BLOCK_N, D, False)

    # Diagonal / edge blocks with masking
    hi = tl.minimum((pid_m + 1) * BLOCK_M, N)
    for start_n in range(full_end, hi, BLOCK_N):
        acc, m_i, l_i = _step(q, acc, m_i, l_i, K, V, base, offs_m, offs_d,
                              start_n, N, qk_scale, BLOCK_N, D, True)

    acc = acc / l_i[:, None]
    o_ptrs = O + base + offs_m[:, None] * D + offs_d[None, :]
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=offs_m[:, None] < N)


def forward(q, k, v):
    B, H, N, D = q.shape
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    o = torch.empty_like(q)
    if N >= 64:
        BLOCK_M, BLOCK_N = 64, 32
    else:
        BLOCK_M, BLOCK_N = 32, 32
    qk_scale = (D ** -0.5) * 1.4426950408889634
    grid = (triton.cdiv(N, BLOCK_M), B * H)
    _attn_fwd[grid](
        q, k, v, o, N, qk_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=D,
        num_warps=4, num_stages=2,
    )
    return o
