# Known-good: flash-style causal attention (online softmax, scores never hit memory).
import torch
import triton
import triton.language as tl


@triton.jit
def _kernel(Q, K, V, O, S, scale, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, PREC: tl.constexpr):
    pm, bh = tl.program_id(0), tl.program_id(1)
    base = bh * S * D
    rm = pm * BM + tl.arange(0, BM)
    rd = tl.arange(0, D)
    q = tl.load(Q + base + rm[:, None] * D + rd[None, :], mask=rm[:, None] < S, other=0.0)
    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    hi = tl.minimum((pm + 1) * BM, S)
    for start in range(0, hi, BN):
        rn = start + tl.arange(0, BN)
        k = tl.load(K + base + rn[None, :] * D + rd[:, None], mask=rn[None, :] < S, other=0.0)
        s = tl.dot(q, k, input_precision=PREC) * scale
        s = tl.where((rm[:, None] >= rn[None, :]) & (rn[None, :] < S), s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(V + base + rn[:, None] * D + rd[None, :], mask=rn[:, None] < S, other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision=PREC)
        m_i = m_new
    tl.store(O + base + rm[:, None] * D + rd[None, :], (acc / l_i[:, None]).to(O.dtype.element_ty),
             mask=rm[:, None] < S)


def forward(q, k, v):
    B, H, S, D = q.shape
    out = torch.empty_like(q)
    prec = "ieee" if q.dtype == torch.float32 else "tf32"
    _kernel[(triton.cdiv(S, 64), B * H)](q.contiguous(), k.contiguous(), v.contiguous(), out, S, D ** -0.5,
                                        D=D, BM=64, BN=64, PREC=prec)
    return out
