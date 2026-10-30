# Deliberately wrong: ignores the dequantization scales. Must fail correctness.
import torch


def forward(a, b, sa, sb):
    return (a.float() @ b.float()).to(sa.dtype)
