from itertools import chain
from os import initgroups

import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from prettytable import PrettyTable
from torch.nn.modules.utils import _pair
from copy import deepcopy

def count_parameters(model, logger):
    table = PrettyTable(["Modules", "Parameters", "Trainable?"])
    total_params = 0
    for name, parameter in model.named_parameters():
        param = parameter.numel()
        table.add_row([name, param, parameter.requires_grad])
        total_params += param
    logger.info(f"\n{table}")
    logger.info(f"Total Trainable Params: {total_params}")
    return total_params

def recursive_traverse(module, layers):
    children = list(module.children())
    if len(children) > 0:
        for child in children:
            recursive_traverse(child, layers)
    else:
        layers.append(module)

def bd_bound(pos_mean, pos_std):
    # Formula from this paper https://www.mdpi.com/1099-4300/19/7/361/htm
    dist = D.Normal(pos_mean, pos_std)
    cond_entropy = dist.entropy().mean(0).sum()
    posmeanflat = pos_mean.flatten(1)
    pairwise_mean_dis = posmeanflat.unsqueeze(1) - posmeanflat.unsqueeze(0)
    posstdflat = pos_std.flatten(1)
    posvar = posstdflat.square()
    pairwise_var_sum = posvar.unsqueeze(1) + posvar.unsqueeze(0)
    logstd = posstdflat.log().sum(1)
    pairwise_std_logprod = logstd.unsqueeze(1) + logstd.unsqueeze(0)
    first_dist = (pairwise_mean_dis.square()/pairwise_var_sum).sum(2)/16
    second_dist = 0.5*(torch.log(0.5*pairwise_var_sum).sum(2)-pairwise_std_logprod)
    pairwise_dist = first_dist + second_dist
    log_dist = torch.logsumexp(-pairwise_dist, dim=1) - torch.log(torch.tensor(pos_mean.size(0), dtype=torch.float32, device=pos_mean.device))
    second_part = log_dist.mean(0)
    entropy = cond_entropy - second_part
    return entropy

def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, nn.BatchNorm2d):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, nn.BatchNorm2d) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)

class StoModel(object):
    def sto_init(self, n_components):
        self.n_components = n_components
        self.sto_modules = [
            m for m in self.modules() if isinstance(m, (StoLinear, StoConv2d))
        ]

    def kl_and_entropy(self, kl_type, entropy_type):
        kl_and_entropy = [
            m.kl_and_entropy(kl_type, entropy_type) for m in self.sto_modules
        ]
        return sum(x[0] for x in kl_and_entropy), sum(x[1] for x in kl_and_entropy)

    def vi_loss(self, x, y, n_sample, kl_type, entropy_type):
        y = y.unsqueeze(1).expand(-1, n_sample)
        logp = D.Categorical(logits=self.forward(x, n_sample)).log_prob(y).mean()
        return (-logp, *self.kl_and_entropy(kl_type, entropy_type))
    
    def entropy(self, n_sample, with_weight):
        return sum(m.entropy(n_sample, with_weight) for m in self.sto_modules)
    
    def nll(self, x, y, n_sample):
        indices = torch.empty(x.size(0)*n_sample, dtype=torch.long, device=x.device)
        prob = torch.cat([self.forward(x, n_sample, indices=torch.full((x.size(0)*n_sample,), idx, out=indices, device=x.device, dtype=torch.long)) for idx in range(self.n_components)], dim=1)
        logp = D.Categorical(logits=prob).log_prob(y.unsqueeze(1).expand(-1, self.n_components*n_sample))
        logp = torch.logsumexp(logp, 1) - torch.log(torch.tensor(self.n_components*n_sample, dtype=torch.float32, device=x.device))
        return -logp.mean(), prob

class StoLayer(object):
    @classmethod
    def convert_deterministic(cls, sto_model, index, det_model, sample=False):
        param_tensors = []
        buffer_tensors = []
        layers = []
        recursive_traverse(sto_model, layers)
        for module in layers:
            if isinstance(module, StoLayer):
                module = module.to_det_module(index, sample)
            param_tensors.extend(module.parameters())
            buffer_tensors.extend(module.buffers())
        for p1, p2 in zip(det_model.parameters(), param_tensors):
            p1.data = p2.data
        for p1, p2 in zip(det_model.buffers(), buffer_tensors):
            p1.data = p2.data
        return det_model
    
    @staticmethod
    def get_mask(mean, std, index, sample):
        if index == 'ones':
            return torch.ones(mean.shape[1:], device=mean.device)
        if index == 'mean':
            return mean.mean(dim=0)
        if sample:
            return D.Normal(mean[index], std[index]).sample()
        return mean[index]
    
    def to_det_module(self, index):
        raise NotImplementedError()

    def sto_init(self, n_components, prior_mean, prior_std, posterior_mean_init=(1.0, 0.5), posterior_std_init=(0.05, 0.02), mode='in'):
        # [1, In, 1, 1]
        shape = [1] * (self.weight.ndim-1)
        if mode == 'in':
            shape[0] = self.weight.shape[1]
        elif mode == 'out':
            shape[0] = self.weight.shape[0]
        elif mode == 'inout':
            shape[0] = sum(self.weight.shape[:2])
        self.posterior_U_mean = nn.Parameter(torch.ones((n_components, *shape)), requires_grad=True)
        self.posterior_U_std = nn.Parameter(torch.ones((n_components, *shape)), requires_grad=True)
        nn.init.normal_(self.posterior_U_std, posterior_std_init[0], posterior_std_init[1])
        nn.init.normal_(self.posterior_U_mean, posterior_mean_init[0], posterior_mean_init[1])
        self.posterior_U_std.data.abs_().expm1_().log_()
        self.mode = mode

        self.prior_mean = nn.Parameter(torch.tensor(prior_mean), requires_grad=False)
        self.prior_std = nn.Parameter(torch.tensor(prior_std), requires_grad=False)
        self.posterior_mean_init = posterior_mean_init
        self.posterior_std_init = posterior_std_init
    
    def get_mult_noise(self, input, indices):
        if indices == 'ones':
            return 1.0
        mean = self.posterior_U_mean
        std = F.softplus(self.posterior_U_std)
        noise = torch.randn((indices.size(0), *mean.shape[1:]), device=input.device, dtype=input.dtype)
        samples = mean[indices] + std[indices] * noise
        return samples
    
    def kl_and_entropy(self, kl_type, entropy_type):
        kl, entropy = self._kl_and_entropy(self.posterior_U_mean, self.posterior_U_std, kl_type, entropy_type)
        return kl, entropy
    
    def _kl_and_entropy(self, pos_mean, pos_std, type='mean', entropy_type='conditional'):
        prior = D.Normal(self.prior_mean, self.prior_std)
        if type == 'mean':
            mean = pos_mean.mean(dim=0)
            std = F.softplus(pos_std).square().sum(0).sqrt() / pos_std.size(0)
            components = D.Normal(mean, std)
            kl = D.kl_divergence(components, prior).sum()
        elif type == 'full':
            pos_std = F.softplus(pos_std)
            dist = D.Normal(pos_mean, pos_std)
            kl = D.kl_divergence(dist, prior).sum()
        elif type == 'upper_bound':
            pos_std = F.softplus(pos_std)
            dist = D.Normal(pos_mean, pos_std)
            entropy = bd_bound(pos_mean, pos_std)
            cross_entropy = (D.kl_divergence(D.Normal(pos_mean, pos_std), prior) + D.Normal(pos_mean, pos_std).entropy()).flatten(1).sum(1).mean()
            kl = cross_entropy - entropy # *pos_std.size(0)
        if entropy_type == 'conditional':
            entropy = dist.entropy().mean(0).sum()
        elif entropy_type == 'conditional_sum':
            entropy = dist.entropy().sum()
        elif entropy_type == 'BD':
            entropy = bd_bound(pos_mean, pos_std)
        return kl, entropy
            
    def _entropy(self, means, stds, n_samples=10000):
        dist = D.MixtureSameFamily(
            mixture_distribution=D.Categorical(torch.ones((means.size(0),), device=means.device)),
            component_distribution=D.Independent(D.Normal(means, stds), means.ndim-1)
        )
        return -dist.log_prob(dist.sample((n_samples,))).mean()
    
    def entropy(self, n_samples=10000, with_weight=True):
        if with_weight:
            return self._entropy(
                self.posterior_U_mean.unsqueeze(1 if self.mode == 'in' else 2) * self.weight, 
                F.softplus(self.posterior_U_std).unsqueeze(1 if self.mode == 'in' else 2) * self.weight.abs() + 1e-10, n_samples) #+ \
        else:
            return self._entropy(self.posterior_U_mean, F.softplus(self.posterior_U_std), n_samples) #+ \
    
    def sto_extra_repr(self):
        return f"n_components={self.posterior_U_mean.size(0)}, prior_mean={self.prior_mean.data.item()}, prior_std={self.prior_std.data.item()}, posterior_mean_init={self.posterior_mean_init}, posterior_std_init={self.posterior_std_init}, mode={self.mode}"

class StoIdentity(nn.Module, StoLayer):
    def __init__(
        self, in_features, n_components=8, prior_mean=1.0, prior_std=0.1, posterior_mean_init=(1.0, 0.5), posterior_std_init=(0.05, 0.02)
    ):
        super(StoIdentity, self).__init__()
        self.bias = None
        self.sto_init(in_features, n_components, prior_mean, prior_std, posterior_mean_init, posterior_std_init)
    
    def forward(self, x, indices):
        return self.mult_noise(x, indices)
    
    def extra_repr(self):
        return f"{self.sto_extra_repr()}"

class StoConv2d(nn.Conv2d, StoLayer):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, groups = 1, bias = True, padding_mode = 'zeros',
        n_components=8, prior_mean=1.0, prior_std=0.1, posterior_mean_init=(1.0, 0.5), posterior_std_init=(0.05, 0.02), mode='in'
    ):
        super(StoConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.sto_init(n_components, prior_mean, prior_std, posterior_mean_init, posterior_std_init, mode)
    
    def forward(self, x, indices):
        noise = self.get_mult_noise(x, indices)
        if 'in' in self.mode:
            x = x * noise[:, :x.shape[1]]
        x = super().forward(x)
        if 'out' in self.mode:
            x = x * noise[:, -x.shape[1]:]
        return x
    
    def to_det_module(self, index, sample=False):
        new_module = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias is not None, self.padding_mode)
        U_mask = StoLayer.get_mask(self.posterior_U_mean, F.softplus(self.posterior_U_std), index, sample)
        new_module.weight.data = self.weight.data.clone()
        if 'in' in self.mode:
            new_module.weight.data *= U_mask[:self.weight.shape[1]]
        if 'out' in self.mode:
            new_module.weight.data *= U_mask[-self.weight.shape[0]:].unsqueeze(1)
        if self.bias is not None:
            new_module.bias.data = self.bias.data.clone()
            if 'out' in self.mode:
                new_module.bias.data *= U_mask[-self.weight.shape[0]:].view(-1)
        return new_module

    def extra_repr(self):
        return f"{super(StoConv2d, self).extra_repr()}, {self.sto_extra_repr()}"

class StoLinear(nn.Linear, StoLayer):
    def __init__(
        self, in_features: int, out_features: int, bias: bool = True,
        n_components=8, prior_mean=1.0, prior_std=0.1, posterior_mean_init=(1.0, 0.5), posterior_std_init=(0.05, 0.02), mode='in'
    ):
        super(StoLinear, self).__init__(in_features, out_features, bias)
        self.sto_init(n_components, prior_mean, prior_std, posterior_mean_init, posterior_std_init, mode)

    def forward(self, x, indices):
        noise = self.get_mult_noise(x, indices)
        if 'in' in self.mode:
            x = x * noise[:, :x.shape[1]]
        x = super().forward(x)
        if 'out' in self.mode:
            x = x * noise[:, -x.shape[1]:]
        return x

    def to_det_module(self, index, sample):
        new_module = nn.Linear(self.in_features, self.out_features, self.bias is not None)
        U_mask = StoLayer.get_mask(self.posterior_U_mean, F.softplus(self.posterior_U_std), index, sample)
        new_module.weight.data = self.weight.data.clone()
        if 'in' in self.mode:
            new_module.weight.data *= U_mask[:self.weight.shape[1]]
        if 'out' in self.mode:
            new_module.weight.data *= U_mask[-self.weight.shape[0]:].unsqueeze(1)
        if self.bias is not None:
            new_module.bias.data = self.bias.data.clone()
            if 'out' in self.mode:
                new_module.bias.data *= U_mask[-self.weight.shape[0]:].view(-1)
        return new_module

    def extra_repr(self):
        return f"{super(StoLinear, self).extra_repr()}, {self.sto_extra_repr()}"



class ECELoss(nn.Module):
    """
    Ported from https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py

    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, probs, labels):
        confidences, predictions = torch.max(probs, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=probs.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * \
                confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin -
                                 accuracy_in_bin) * prop_in_bin

        return ece
