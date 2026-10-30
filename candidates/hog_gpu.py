# Allocates far more GPU memory than the cap allows; must come back as a failed report.
import torch


def forward(x):
    torch.empty(int(40 * 2**30) // 4, dtype=torch.float32, device="cuda")
    return x.clone()
