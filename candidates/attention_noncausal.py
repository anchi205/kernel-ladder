# Deliberately wrong: forgets the causal mask. Must fail correctness.
import torch


def forward(q, k, v):
    s = (q @ k.transpose(-1, -2)) * (q.shape[-1] ** -0.5)
    e = torch.exp(s - s.max(-1, keepdim=True).values)
    return (e / e.sum(-1, keepdim=True)) @ v
