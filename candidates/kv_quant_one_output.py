# Deliberately wrong: returns only the int8 tensor, dropping the scales.
import torch


def forward(x):
    xf = x.float()
    scale = xf.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 127
    return torch.round(xf / scale).clamp(-127, 127).to(torch.int8)
