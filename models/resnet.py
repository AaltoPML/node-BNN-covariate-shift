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

__all__ = ["DetResNet18", "StoResNet18", "DetResNet9", "StoResNet9"]


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def sto_conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, n_components=4, prior_mean=1.0, prior_std=0.1, posterior_mean_init=(1.0, 0.05), posterior_std_init=(0.75, 0.02), mode='in') -> nn.Conv2d:
    """3x3 convolution with padding"""
    return StoConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation,
                     n_components=n_components, prior_mean=prior_mean, prior_std=prior_std,
                     posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, mode=mode)


def sto_conv1x1(in_planes: int, out_planes: int, stride: int = 1, n_components=4, prior_mean=1.0, prior_std=0.1, posterior_mean_init=(1.0, 0.05), posterior_std_init=(0.75, 0.02), mode='in') -> nn.Conv2d:
    """1x1 convolution"""
    return StoConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False,
                     n_components=n_components, prior_mean=prior_mean, prior_std=prior_std,
                     posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, mode=mode)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class StoBasicBlock(nn.Module, StoLayer):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        n_components=4,
        prior_mean=1.0, 
        prior_std=0.1, 
        posterior_mean_init=(1.0, 0.05), 
        posterior_std_init=(0.75, 0.02), mode='in'
    ) -> None:
        super(StoBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = sto_conv3x3(inplanes, planes, stride, n_components=n_components, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, mode=mode)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = sto_conv3x3(planes, planes, n_components=n_components, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, mode=mode)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor, indices: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x, indices)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, indices)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x, indices)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class StoBottleneck(nn.Module, StoLayer):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        n_components=4,
        prior_mean=1.0, 
        prior_std=0.1, 
        posterior_mean_init=(1.0, 0.05), 
        posterior_std_init=(0.75, 0.02)
    ) -> None:
        super(StoBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, n_components=n_components, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation, n_components=n_components, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion, n_components=n_components, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor, indices: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x, indices)
        if isinstance(self.bn1, StoBatchNorm2d):
            out = self.bn1(out, indices)
        else:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, indices)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out, indices)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x, indices)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        # self.normalize_input = torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = conv3x3(3, self.inplanes)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AvgPool2d((4, 4))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    # type: ignore[arg-type]
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    # type: ignore[arg-type]
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        # x = self.normalize_input(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.log_softmax(x, -1)

        return x

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

class StoResNet(nn.Module, StoModel):

    def __init__(
        self,
        block: Type[Union[StoBasicBlock, StoBottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        n_components=4,
        prior_mean=1.0, 
        prior_std=0.1, 
        posterior_mean_init=(1.0, 0.05), 
        posterior_std_init=(0.75, 0.02),
        bn_momentum=0.1,
        mode='in'
    ) -> None:
        super(StoResNet, self).__init__()
        # self.normalize_input = torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, momentum=bn_momentum)
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = sto_conv3x3(3, self.inplanes, n_components=n_components, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, mode=mode)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], n_components=n_components, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, mode=mode)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], n_components=n_components, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, mode=mode)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], n_components=n_components, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, mode=mode)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], n_components=n_components, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, mode=mode)
        self.avgpool = nn.AvgPool2d((4, 4))
        self.fc = StoLinear(512 * block.expansion, num_classes, n_components=n_components, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, mode=mode)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, StoBottleneck):
                    # type: ignore[arg-type]
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, StoBasicBlock):
                    # type: ignore[arg-type]
                    nn.init.constant_(m.bn2.weight, 0)
        self.sto_init(n_components)

    def _make_layer(self, block: Type[Union[StoBasicBlock, StoBottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False, n_components=4, prior_mean=1.0, prior_std=0.1, posterior_mean_init=(1.0, 0.05), 
                    posterior_std_init=(0.75, 0.02), mode='in') -> StoSequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = StoSequential(
                sto_conv1x1(self.inplanes, planes * block.expansion, stride, n_components=n_components, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, mode=mode),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, n_components=n_components, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, mode=mode))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation, mode=mode,
                                norm_layer=norm_layer, n_components=n_components, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init))

        return StoSequential(*layers)

    def forward(self, x, L=1, indices=None, return_kl=False):
        # x = self.normalize_input(x)
        if L > 1:
            x = torch.repeat_interleave(x, L, dim=0)
        if indices is None:
            indices = torch.arange(x.size(0), dtype=torch.long, device=x.device) % self.n_components
        x = self.conv1(x, indices)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x, indices)
        x = self.layer2(x, indices)
        x = self.layer3(x, indices)
        x = self.layer4(x, indices)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x, indices)
        x = F.log_softmax(x, -1)
        x = x.view(-1, L, x.size(1))

        return x

def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    return model

def _storesnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    **kwargs: Any
) -> ResNet:
    model = StoResNet(block, layers, **kwargs)
    return model

def resnet18(**kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2],
                   **kwargs)

class DetResNet18(ResNet):
    def __init__(self, n_classes):
        super(DetResNet18, self).__init__(
            BasicBlock, [2, 2, 2, 2], n_classes
        )

class StoResNet18(StoResNet):
    def __init__(self, n_classes, n_components, prior_mean, prior_std, posterior_mean_init, posterior_std_init, bn_momentum=0.1, bn_layer='det', mode='in'):
        super(StoResNet18, self).__init__(
            StoBasicBlock, [2, 2, 2, 2], n_classes, n_components=n_components, prior_mean=prior_mean, prior_std=prior_std,
            posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, bn_momentum=bn_momentum,
            norm_layer=nn.BatchNorm2d,
            mode = mode
        )

class DetResNet9(ResNet):
    def __init__(self, n_classes):
        super(DetResNet9, self).__init__(
            BasicBlock, [1, 1, 1, 1], n_classes
        )

class StoResNet9(StoResNet):
    def __init__(self, n_classes, n_components, prior_mean, prior_std, posterior_mean_init, posterior_std_init, bn_momentum=0.1, bn_layer='det', mode='in'):
        super(StoResNet9, self).__init__(
            StoBasicBlock, [1, 1, 1, 1], n_classes, n_components=n_components, prior_mean=prior_mean, prior_std=prior_std,
            posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, bn_momentum=bn_momentum,
            norm_layer=nn.BatchNorm2d,
            mode = mode
        )
