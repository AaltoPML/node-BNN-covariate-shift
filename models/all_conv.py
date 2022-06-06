import math
from typing import Any, Callable, List, Optional, Type, Union
from functools import partial
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
from torch import Tensor

from .utils import (StoConv2d, StoIdentity, StoModel,
                    StoLayer, StoLinear)

__all__ = ["StoAllConv", "DetAllConv"]


class StoAllConv(nn.Module, StoModel):
    def __init__(self, n_classes, n_components, prior_mean=1.0, prior_std=0.1, posterior_mean_init=(1.0, 0.05), posterior_std_init=(0.75, 0.02), mode='in'):
        super(StoAllConv, self).__init__()
        self.normalize_input = torchvision.transforms.Normalize((0.0, 0.0, 0.0), (0.2023/0.2470, 0.1994/0.2435, 0.2010/0.2616))
        self.conv1 = StoConv2d(
            3, 96, kernel_size=3, stride=1, padding=1, n_components=n_components, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init,
            posterior_std_init=posterior_std_init, mode=mode
        )
        self.conv2 = StoConv2d(
            96, 96, kernel_size=3, stride=1, padding=1, n_components=n_components, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init,
            posterior_std_init=posterior_std_init, mode=mode
        )
        self.conv3 = StoConv2d(
            96, 96, kernel_size=3, stride=2, padding=1, n_components=n_components, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init,
            posterior_std_init=posterior_std_init, mode=mode
        )
        self.conv4 = StoConv2d(
            96, 192, kernel_size=3, stride=1, padding=1, n_components=n_components, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init,
            posterior_std_init=posterior_std_init, mode=mode
        )
        self.conv5 = StoConv2d(
            192, 192, kernel_size=3, stride=1, padding=1, n_components=n_components, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init,
            posterior_std_init=posterior_std_init, mode=mode
        )
        self.conv6 = StoConv2d(
            192, 192, kernel_size=3, stride=2, padding=1, n_components=n_components, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init,
            posterior_std_init=posterior_std_init, mode=mode
        )
        self.conv7 = StoConv2d(
            192, 192, kernel_size=3, stride=1, padding=1, n_components=n_components, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init,
            posterior_std_init=posterior_std_init, mode=mode
        )
        self.conv8 = StoConv2d(
            192, 192, kernel_size=1, stride=1, padding=0, n_components=n_components, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init,
            posterior_std_init=posterior_std_init, mode=mode
        )
        self.conv8 = StoConv2d(
            192, n_classes, kernel_size=1, stride=1, padding=0, n_components=n_components, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init,
            posterior_std_init=posterior_std_init, mode=mode
        )

        self.sto_init(n_components)
    
    def forward(self, x, L=1, indices=None, log_softmax=True):
        x = self.normalize_input(x)
        if L > 1:
            x = torch.repeat_interleave(x, L, dim=0)
        if indices is None:
            indices = torch.arange(x.size(0), dtype=torch.long, device=x.device) % self.n_components
        x = self.conv1(x, indices)
        x = F.relu(x, True)
        x = self.conv2(x, indices)
        x = F.relu(x, True)
        x = self.conv3(x, indices)
        x = F.relu(x, True)
        x = self.conv4(x, indices)
        x = F.relu(x, True)
        x = self.conv5(x, indices)
        x = F.relu(x, True)
        x = self.conv6(x, indices)
        x = F.relu(x, True)
        x = self.conv7(x, indices)
        x = F.relu(x, True)
        x = self.conv8(x, indices)
        x = F.relu(x, True)
        x = F.avg_pool2d(x, 8)
        x = torch.flatten(x, 1)
        if log_softmax:
            x = F.log_softmax(x, -1)
        x = x.view(-1, L, x.size(1))

        return x

class DetAllConv(nn.Module):
    def __init__(self, n_classes):
        super(DetAllConv, self).__init__()
        self.normalize_input = torchvision.transforms.Normalize((0.0, 0.0, 0.0), (0.2023/0.2470, 0.1994/0.2435, 0.2010/0.2616))
        self.conv1 = nn.Conv2d(
            3, 96, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            96, 96, kernel_size=3, stride=1, padding=1
        )
        self.conv3 = nn.Conv2d(
            96, 96, kernel_size=3, stride=2, padding=1
        )
        self.conv4 = nn.Conv2d(
            96, 192, kernel_size=3, stride=1, padding=1
        )
        self.conv5 = nn.Conv2d(
            192, 192, kernel_size=3, stride=1, padding=1
        )
        self.conv6 = nn.Conv2d(
            192, 192, kernel_size=3, stride=2, padding=1
        )
        self.conv7 = nn.Conv2d(
            192, 192, kernel_size=3, stride=1, padding=1
        )
        self.conv8 = nn.Conv2d(
            192, 192, kernel_size=1, stride=1, padding=0
        )
        self.conv8 = nn.Conv2d(
            192, n_classes, kernel_size=1, stride=1, padding=0
        )
    
    def forward(self, x, log_softmax=True):
        x = self.normalize_input(x)
        x = self.conv1(x)
        x = F.relu(x, True)
        x = self.conv2(x)
        x = F.relu(x, True)
        x = self.conv3(x)
        x = F.relu(x, True)
        x = self.conv4(x)
        x = F.relu(x, True)
        x = self.conv5(x)
        x = F.relu(x, True)
        x = self.conv6(x)
        x = F.relu(x, True)
        x = self.conv7(x)
        x = F.relu(x, True)
        x = self.conv8(x)
        x = F.relu(x, True)
        x = F.avg_pool2d(x, 8)
        x = torch.flatten(x, 1)
        if log_softmax:
            x = F.log_softmax(x, -1)
        return x
