# Deliberately wrong: no max-subtraction, overflows on large inputs. The check must fail it.
import torch


def forward(x):
    e = torch.exp(x)
    return e / e.sum(-1, keepdim=True)
