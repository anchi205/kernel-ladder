# Touches far more host memory than the cap allows; the watchdog must kill the worker.
import torch


def forward(x):
    torch.ones(int(8 * 2**30) // 4, dtype=torch.float32)
    return x.clone()
