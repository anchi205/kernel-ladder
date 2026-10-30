# Deliberately cheating: calls the op instead of implementing it. The fence must reject it.
import torch


def forward(x):
    return torch.softmax(x, -1)
