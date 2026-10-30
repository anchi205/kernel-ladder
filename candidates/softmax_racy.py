# Deliberately nondeterministic on large inputs, with noise inside the correctness tolerance,
# so only the determinism check can catch it.
import torch


def forward(x):
    e = torch.exp(x - x.max(-1, keepdim=True).values)
    out = e / e.sum(-1, keepdim=True)
    if x.numel() > 1 << 24:
        out = out * (1 + 1e-5 * torch.rand_like(out))
    return out
