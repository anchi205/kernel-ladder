# Deliberately wrong: scales its input in place. Must be caught by the input-mutation check.
import torch


def forward(x):
    x.mul_(1.0001)
    xf = x.float()
    scale = xf.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 127
    return torch.round(xf / scale).clamp(-127, 127).to(torch.int8), scale.squeeze(-1)
