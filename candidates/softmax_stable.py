import torch


def forward(x):
    e = torch.exp(x - x.max(-1, keepdim=True).values)
    return e / e.sum(-1, keepdim=True)
