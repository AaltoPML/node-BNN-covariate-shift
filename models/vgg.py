"""
    VGG model definition
    ported from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
"""

import math

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

from .utils import StoLayer, StoLinear, StoConv2d, StoModel

__all__ = ["DetVGG16", "DetVGG16BN", "DetVGG19", "DetVGG19BN", "StoVGG16", "StoVGG16BN", "StoVGG16CBN", "StoVGG19", "StoVGG19BN"]


def make_layers(cfg, batch_norm=False):
    layers = list()
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.ModuleList(layers)

def make_sto_layers(cfg, batch_norm=False, n_components=2, prior_mean=1.0, prior_std=1.0, posterior_mean_init=(1.0, 0.75), posterior_std_init=(0.05, 0.02), mode='in'):
    layers = list()
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = StoConv2d(in_channels, v, kernel_size=3, padding=1,
                               n_components=n_components, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, mode=mode)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.ModuleList(layers)


cfg = {
    16: [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    19: [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGG(nn.Module):
    def __init__(self, num_classes=10, depth=16, batch_norm=False, dropout=True):
        super(VGG, self).__init__()
        # self.normalize_input = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.features = make_layers(cfg[depth], batch_norm)
        self.classifier = nn.ModuleList([
            *((nn.Dropout(),
            nn.Linear(512, 512))  if dropout else (nn.Linear(512, 512),)),
            nn.ReLU(True),
            *((nn.Dropout(),
            nn.Linear(512, 512))  if dropout else (nn.Linear(512, 512),)),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        ])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.bias.data.zero_()

    def forward(self, x):
        # x = self.normalize_input(x)
        for layer in self.features:
            x = layer(x)
        x = x.view(x.size(0), -1)
        for layer in self.classifier:
            x = layer(x)
        x = F.log_softmax(x, dim=-1)
        return x

class StoVGG(nn.Module, StoModel):
    def __init__(self, num_classes=10, depth=16, n_components=2, prior_mean=1.0, prior_std=1.0, posterior_mean_init=(1.0, 0.75), posterior_std_init=(0.05, 0.02), batch_norm=False, cls_batch_norm=False, mode='in'):
        super(StoVGG, self).__init__()
        # self.normalize_input = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.features = make_sto_layers(cfg[depth], batch_norm, n_components, prior_mean, prior_std, posterior_mean_init, posterior_std_init, mode)
        self.classifier = nn.ModuleList([
            StoLinear(512, 512, n_components=n_components, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, mode=mode),
            nn.Sequential(nn.BatchNorm1d(512), nn.ReLU(True)) if cls_batch_norm else nn.ReLU(True),
            StoLinear(512, 512, n_components=n_components, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, mode=mode),
            nn.Sequential(nn.BatchNorm1d(512), nn.ReLU(True)) if cls_batch_norm else nn.ReLU(True),
            StoLinear(512, num_classes, n_components=n_components, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, mode=mode),
#            StoIdentity((num_classes, ), n_components=n_components, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init)
        ])
        self.sto_init(n_components)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.bias.data.zero_()

    def forward(self, x, L=1, indices=None):
        # x = self.normalize_input(x)
        if L > 1:
            x = torch.repeat_interleave(x, L, dim=0)
        if indices is None:
            indices = torch.arange(x.size(0), dtype=torch.long, device=x.device) % self.n_components
        for layer in self.features:
            if isinstance(layer, StoLayer):
                x = layer(x, indices)
            else:
                x = layer(x)
        x = x.view(x.size(0), -1)
        for layer in self.classifier:
            if isinstance(layer, StoLayer):
                x = layer(x, indices)
            else:
                x = layer(x)
        x = F.log_softmax(x, -1)
        x = x.view(-1, L, x.size(1))
        return x

class DetVGG16(VGG):
    def __init__(self, num_classes, dropout=True):
        super(DetVGG16, self).__init__(num_classes, 16, False, dropout)


class DetVGG16BN(VGG):
    def __init__(self, num_classes):
        super(DetVGG16BN, self).__init__(num_classes, 16, True)


class DetVGG19(VGG):
    def __init__(self, num_classes):
        super(DetVGG19, self).__init__(num_classes, 19, False)


class DetVGG19BN(VGG):
    def __init__(self, num_classes):
        super(DetVGG19BN, self).__init__(num_classes, 19, True)

class StoVGG16(StoVGG):
    def __init__(self, num_classes, n_components, prior_mean, prior_std, posterior_mean_init, posterior_std_init, mode):
        super(StoVGG16, self).__init__(num_classes, 16, n_components, prior_mean, prior_std, posterior_mean_init, posterior_std_init, False, False, mode)


class StoVGG16BN(StoVGG):
    def __init__(self, num_classes, n_components, prior_mean, prior_std, posterior_mean_init, posterior_std_init, mode):
        super(StoVGG16BN, self).__init__(num_classes, 16, n_components, prior_mean, prior_std, posterior_mean_init, posterior_std_init, True, False, mode)

class StoVGG16CBN(StoVGG):
    def __init__(self, num_classes, n_components, prior_mean, prior_std, posterior_mean_init, posterior_std_init, mode):
        super(StoVGG16CBN, self).__init__(num_classes, 16, n_components, prior_mean, prior_std, posterior_mean_init, posterior_std_init, True, True, mode)

class StoVGG19(StoVGG):
    def __init__(self, num_classes, n_components, prior_mean, prior_std, posterior_mean_init, posterior_std_init):
        super(StoVGG19, self).__init__(num_classes, 19, n_components, prior_mean, prior_std, posterior_mean_init, posterior_std_init, False)


class StoVGG19BN(StoVGG):
    def __init__(self, num_classes, n_components, prior_mean, prior_std, posterior_mean_init, posterior_std_init):
        super(StoVGG19BN, self).__init__(num_classes, 19, n_components, prior_mean, prior_std, posterior_mean_init, posterior_std_init, True)
