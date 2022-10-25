import numpy as np
import math
import torch
from torch import nn
import torchvision
from torch.nn import functional as F

import utils.framework.util_function as uf
import model.framework.weight_init as wi
from utils.util_class import MyExceptionToCatch
import RIDet3DAddon.torch.config as cfg3d


def sigmoid_with_margin(x, delta=cfg3d.Architecture.SIGMOID_DELTA, low=0, high=1):
    y = torch.sigmoid(x)
    z = (high - low + 2 * delta) * y + low - delta
    return z


def inv_sigmoid_with_margin(z, delta=cfg3d.Architecture.SIGMOID_DELTA, low=0, high=1, eps=1e-7):
    y = (z - low + delta) / (high - low + 2 * delta)
    # assert torch.all(torch.logical_and(z > low - delta, z < high + delta))
    # assert torch.all(torch.logical_and(y >= 0, y <= 1))
    x = torch.clip(y, eps, 1 - eps)
    x = x / (1 - x)
    x = torch.log(x)
    return x


class CustomConv2D(nn.Module):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support zero-size tensor and more features.
    """

    def __init__(self, *args, **kwargs):

        """
        in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
        bias=True, padding_mode='zeros', device=None, dtype=None
        """
        super().__init__()
        act_name = kwargs.pop("activation", "relu")
        bn_name = kwargs.pop("bn", True)
        scope = kwargs.pop("scope", None)
        self.device = cfg3d.Train.DEVICE
        self.conv = nn.Conv2d(*args,  **kwargs, device=self.device)
        self.bn = nn.BatchNorm2d(self.conv.out_channels, device=self.device) if bn_name else None
        self.activation = select_activation(act_name)
        self.out_channels = self.conv.out_channels
        # if scope == "back":
        #     wi.c2_msra_fill(self.conv)

    def forward(self, x):
        if not torch.jit.is_scripting():
            if x.numel() == 0 and self.training:
                assert not isinstance(
                    self.norm, torch.nn.SyncBatchNorm
                ), "SyncBatchNorm does not support empty inputs!"

        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


def select_activation(activation):
    if activation == "leaky_relu":
        act = nn.LeakyReLU()
    elif activation == "mish":
        act = nn.Mish()
    elif activation == "relu":
        act = nn.ReLU()
    elif activation == "swish":
        act = nn.Hardswish()
    elif activation is False:
        act = None
    else:
        raise MyExceptionToCatch(f"[CustomConv2D] invalid activation name: {activation}")
    return act