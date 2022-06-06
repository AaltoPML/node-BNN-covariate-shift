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

from .utils import (StoConv2d, StoIdentity,
                    StoLayer, StoLinear, StoModel)

__all__ = ["DetPreActResNet18", "StoPreActResNet18"]


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class StoSequential(nn.Sequential, StoLayer):
    def __init__(self, *args):
        super(StoSequential, self).__init__(*args)

    def forward(self, input, indices):
        for module in self:
            if isinstance(module, StoLayer):
                input = module(input, indices)
            else:
                input = module(input)
        return input


class StoPreActBlock(nn.Module, StoLayer):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1,
                 n_components=4,
                 prior_mean=1.0,
                 prior_std=0.1,
                 posterior_mean_init=(1.0, 0.05),
                 posterior_std_init=(0.75, 0.02), mode='in'):
        super(StoPreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = StoConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1,
                               bias=False, n_components=n_components, prior_mean=prior_mean, prior_std=prior_std,
                               posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, mode=mode)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = StoConv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False, n_components=n_components, prior_mean=prior_mean, prior_std=prior_std,
                               posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, mode=mode)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = StoSequential(
                StoConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride,
                          bias=False, n_components=n_components, prior_mean=prior_mean, prior_std=prior_std,
                          posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, mode=mode)
            )

    def forward(self, x, indices):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out, indices) if hasattr(
            self, 'shortcut') else x
        out = self.conv1(out, indices)
        out = self.conv2(F.relu(self.bn2(out)), indices)
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, initial_channels, num_classes, stride=1):
        super(PreActResNet, self).__init__()
        self.in_planes = initial_channels
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3,
                               initial_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.layer1 = self._make_layer(
            block, initial_channels * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(
            block, initial_channels * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(
            block, initial_channels * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(
            block, initial_channels * 8, num_blocks[3], stride=2)
        self.linear = nn.Linear(initial_channels * 8 *
                                block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 8)
        out = out.reshape(out.size(0), -1)
        out = self.linear(out)
        return F.log_softmax(out, dim=-1)


class StoPreActResNet(nn.Module, StoModel):
    def __init__(self, block, num_blocks, initial_channels, num_classes, stride=1, n_components=4,
                 prior_mean=1.0,
                 prior_std=0.1,
                 posterior_mean_init=(1.0, 0.05),
                 posterior_std_init=(0.75, 0.02), mode='in'):
        super(StoPreActResNet, self).__init__()
        self.in_planes = initial_channels
        self.num_classes = num_classes
        self.conv1 = StoConv2d(3,
                               initial_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False, n_components=n_components, prior_mean=prior_mean, prior_std=prior_std,
                               posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, mode=mode)
        self.layer1 = self._make_layer(
            block, initial_channels * 1, num_blocks[0], stride=1, n_components=n_components, prior_mean=prior_mean, prior_std=prior_std,
            posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, mode=mode)
        self.layer2 = self._make_layer(
            block, initial_channels * 2, num_blocks[1], stride=2, n_components=n_components, prior_mean=prior_mean, prior_std=prior_std,
            posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, mode=mode)
        self.layer3 = self._make_layer(
            block, initial_channels * 4, num_blocks[2], stride=2, n_components=n_components, prior_mean=prior_mean, prior_std=prior_std,
            posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, mode=mode)
        self.layer4 = self._make_layer(
            block, initial_channels * 8, num_blocks[3], stride=2, n_components=n_components, prior_mean=prior_mean, prior_std=prior_std,
            posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, mode=mode)
        self.linear = StoLinear(initial_channels * 8 * block.expansion, num_classes, bias=True, n_components=n_components, prior_mean=prior_mean, prior_std=prior_std,
                                posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, mode=mode)
        self.sto_init(n_components)

    def _make_layer(self, block, planes, num_blocks, stride, n_components, prior_mean, prior_std, posterior_mean_init, posterior_std_init, mode):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, n_components=n_components, prior_mean=prior_mean, prior_std=prior_std,
                                posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, mode=mode))
            self.in_planes = planes * block.expansion
        return StoSequential(*layers)

    def forward(self, x, L=1, indices=None):
        # x = self.normalize_input(x)
        if L > 1:
            x = torch.repeat_interleave(x, L, dim=0)
        if indices is None:
            indices = torch.arange(
                x.size(0), dtype=torch.long, device=x.device) % self.n_components
        out = self.conv1(x, indices)
        out = self.layer1(out, indices)
        out = self.layer2(out, indices)
        out = self.layer3(out, indices)
        out = self.layer4(out, indices)
        out = F.avg_pool2d(out, 8)
        out = out.reshape(out.size(0), -1)
        out = self.linear(out, indices)
        return F.log_softmax(out, dim=-1).view(-1, L, out.size(1))


def DetPreActResNet18(num_classes=10):
    return PreActResNet(PreActBlock, [2, 2, 2, 2], 64, num_classes, stride=1)


class StoPreActResNet18(StoPreActResNet):
    def __init__(self, n_classes, n_components, prior_mean, prior_std, posterior_mean_init, posterior_std_init, mode):
        super(StoPreActResNet18, self).__init__(
            StoPreActBlock, [2, 2, 2, 2], 64, n_classes, stride=1,
            n_components=n_components, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, mode=mode)
