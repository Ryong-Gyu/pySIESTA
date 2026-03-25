import torch
import functools


def get_optimizer(name, **kwargs):
    return functools.partial(getattr(torch.optim, name), **kwargs)
