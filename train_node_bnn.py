import logging
import os
from bisect import bisect
from itertools import chain

import numpy as np
import torch
import torch.distributions as D
import torchvision
from sacred import Experiment
from sacred.observers import FileStorageObserver, RunObserver
from scipy.stats import entropy
import json
from models import (SAM, DetAllConv,
                    DetResNet18, DetVGG16,
                    ECELoss, StoAllConv, StoResNet18, 
                    StoVGG16, StoVGG16BN, StoVGG16CBN, StoPreActResNet18,
                    count_parameters, disable_running_stats,
                    enable_running_stats)
from new_datasets import get_corrupt_data_loader, get_data_loader

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


class SetID(RunObserver):
    priority = 50  # very high priority

    def started_event(self, ex_info, command, host_info, start_time, config, meta_info, _id):
        return f"{config['model_name']}_{config['seed']}_{config['dataset']}_{config['name']}"


EXPERIMENT = 'experiments'
BASE_DIR = EXPERIMENT
ex = Experiment(EXPERIMENT)
ex.observers.append(SetID())
ex.observers.append(FileStorageObserver(BASE_DIR))


@ex.config
def my_config():
    ece_bins = 15
    seed = 1  # Random seed
    name = 'name'  # Unique name for the folder of the experiment
    model_name = 'StoResNet18'  # Choose with model to train
    # the KL weight will increase from <kl_min> to <kl_max> for <last_iter> iterations.
    kl_weight = {
        'kl_min': 0.0,
        'kl_max': 1.0,
        'last_iter': 200
    }
    batch_size = 128  # Batch size
    test_batch_size = 512
    prior_mean = 1.0  # Mean of the Gaussian prior
    prior_std = 0.5  # Std of the Gaussian prior
    n_components = 2  # Number of components in the posterior
    # Options of the deterministic weights for the SGD
    det_params = {
        'lr': 0.1, 'weight_decay': 5e-4
    }
    # Options of the variational parameters for the SGD
    sto_params = {
        'lr': 0.1, 'weight_decay': 0.0, 'momentum': 0.0, 'nesterov': False
    }
    # Universal options for the SGD
    sgd_params = {
        'momentum': 0.9,
        'dampening': 0.0,
        'nesterov': True
    }
    num_epochs = 300  # Number of training epoch
    validation = True  # Whether of not to use a validation set
    validation_fraction = 0.1  # Size of the validation set
    save_freq = 301  # Frequency of saving checkpoint
    num_train_sample = 1  # Number of samples drawn from each component during training
    num_test_sample = 1  # Number of samples drawn from each component during testing
    logging_freq = 1  # Logging frequency
    device = 'cuda'
    lr_ratio_det = 0.01  # For annealing the learning rate of the deterministic weights
    lr_ratio_sto = 1/3  # For annealing the learning rate of the variational parameters
    # First value chooses which epoch to start decreasing the learning rate and the second value chooses which epoch to stop. See the schedule function for more information.
    det_milestones = (0.5, 0.9)
    sto_milestones = (0.5, 0.9)
    kl_type = 'mean'
    gamma = 1.0
    entropy_type = 'conditional'
    augment_data = True
    if not torch.cuda.is_available():
        device = 'cpu'
    # Mean and std to init the component means in the posterior
    posterior_mean_init = (1.0, 0.5)
    # Mean and std to init the component stds in the posterior
    posterior_std_init = (0.05, 0.02)
    dataset = 'cifar100'  # Dataset of the experiment
    if dataset == 'cifar100' or dataset == 'vgg_cifar100':
        num_classes = 100
    elif dataset == 'cifar10' or dataset == 'vgg_cifar10' or dataset == 'fmnist':
        num_classes = 10
    elif dataset == 'tinyimagenet':
        num_classes = 200

    bn_momentum = 0.1
    noise_mode = 'in'
    det_checkpoint = ''
    num_train_workers = 4
    num_test_workers = 2
    data_norm_stat = None


@ex.capture(prefix='kl_weight')
def get_vi_weight(epoch, kl_min, kl_max, last_iter):
    value = (kl_max-kl_min)/last_iter
    return min(kl_max, kl_min + epoch*value)


def schedule(num_epochs, epoch, milestones, lr_ratio):
    t = epoch / num_epochs
    m1, m2 = milestones
    if t <= m1:
        factor = 1.0
    elif t <= m2:
        factor = 1.0 - (1.0 - lr_ratio) * (t - m1) / (m2 - m1)
    else:
        factor = lr_ratio
    return factor


@ex.capture
def get_model(model_name, num_classes, prior_mean, prior_std, n_components, device, sgd_params, det_params, sto_params, num_epochs, det_milestones, sto_milestones,
              lr_ratio_det, lr_ratio_sto, posterior_mean_init, posterior_std_init, bn_momentum, noise_mode):
    if model_name == 'StoResNet18':
        model = StoResNet18(num_classes, n_components, prior_mean, prior_std,
                            posterior_mean_init, posterior_std_init, bn_momentum, mode=noise_mode)
    elif model_name == 'StoPreActResNet18':
        model = StoPreActResNet18(num_classes, n_components, prior_mean, prior_std,
                                  posterior_mean_init, posterior_std_init, noise_mode)
    elif model_name == 'StoVGG16':
        model = StoVGG16(num_classes, n_components, prior_mean, prior_std,
                         posterior_mean_init, posterior_std_init, mode=noise_mode)
    elif model_name == 'StoVGG16BN':
        model = StoVGG16BN(num_classes, n_components, prior_mean,
                           prior_std, posterior_mean_init, posterior_std_init)
    elif model_name == 'StoVGG16CBN':
        model = StoVGG16CBN(num_classes, n_components, prior_mean, prior_std, posterior_mean_init, posterior_std_init)
    elif model_name == 'StoAllConv':
        model = StoAllConv(num_classes, n_components, prior_mean, prior_std,
                         posterior_mean_init, posterior_std_init, mode = noise_mode)
    detp=[]
    stop=[]
    for name, param in model.named_parameters():
        if 'posterior' in name or 'prior' in name:
            stop.append(param)
        else:
            detp.append(param)
    optimizer=torch.optim.SGD(
        [{
            'params': detp,
            **det_params
        }, {
            'params': stop,
            **sto_params
        }], **sgd_params)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, [lambda e: schedule(
        num_epochs, e, det_milestones, lr_ratio_det), lambda e: schedule(num_epochs, e, sto_milestones, lr_ratio_sto)])

    model.to(device)
    return model, optimizer, scheduler


@ex.capture
def get_dataloader(batch_size, test_batch_size, num_test_sample, validation, validation_fraction, dataset, augment_data, num_train_workers, num_test_workers, data_norm_stat):
    test_bs = test_batch_size // num_test_sample
    return get_data_loader(dataset, train_bs=batch_size, test_bs=test_bs, validation=validation, validation_fraction=validation_fraction,
                           augment = augment_data, num_train_workers = num_train_workers, num_test_workers = num_test_workers, norm_stat=data_norm_stat)

@ex.capture
def get_corruptdataloader(intensity, test_batch_size, num_test_sample, dataset, num_test_workers, data_norm_stat):
    test_bs = test_batch_size // num_test_sample
    return get_corrupt_data_loader(dataset, intensity, batch_size=test_bs, root_dir='data/', num_workers=num_test_workers, norm_stat=data_norm_stat)

@ex.capture
def get_logger(_run, _log):
    fh=logging.FileHandler(os.path.join(BASE_DIR, _run._id, 'train.log'))
    fh.setLevel(logging.INFO)
    formatter=logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    _log.addHandler(fh)
    return _log


@ex.capture
def test_stochastic(model, dataloader, device, num_test_sample, ece_bins):
    tnll = 0
    acc = [0, 0, 0]
    nll_miss = 0
    y_prob = []
    y_true = []
    y_prob_all = []
    model.eval()
    with torch.no_grad():
        for bx, by in dataloader:
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)
            indices = torch.empty(bx.size(0)*num_test_sample, dtype=torch.long, device=bx.device)
            prob = torch.cat([model.forward(bx, num_test_sample, indices=torch.full((bx.size(0)*num_test_sample,), idx, out=indices, device=bx.device, dtype=torch.long)) for idx in range(model.n_components)], dim=1)
            y_target = by.unsqueeze(1).expand(-1, num_test_sample*model.n_components)
            bnll = D.Categorical(logits=prob).log_prob(y_target)
            bnll = torch.logsumexp(bnll, dim=1) - torch.log(torch.tensor(num_test_sample*model.n_components, dtype=torch.float32, device=bnll.device))
            tnll -= bnll.sum().item()
            vote = prob.exp().mean(dim=1)
            top3 = torch.topk(vote, k=3, dim=1, largest=True, sorted=True)[1]
            y_prob_all.append(prob.exp().cpu().numpy())
            y_prob.append(vote.cpu().numpy())
            y_true.append(by.cpu().numpy())
            y_miss = top3[:, 0] != by
            if y_miss.sum().item() > 0:
                nll_miss -= bnll[y_miss].sum().item()
            for k in range(3):
                acc[k] += (top3[:, k] == by).sum().item()
    nll_miss /= len(dataloader.dataset) - acc[0]
    tnll /= len(dataloader.dataset)
    for k in range(3):
        acc[k] /= len(dataloader.dataset)
    acc = np.cumsum(acc)
    y_prob = np.concatenate(y_prob, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    y_prob_all = np.concatenate(y_prob_all, axis=0)
    total_entropy = entropy(y_prob, axis=1)
    aleatoric = entropy(y_prob_all, axis=-1).mean(axis=-1)
    epistemic = total_entropy - aleatoric
    ece = ECELoss(ece_bins)
    ece_val = ece(torch.from_numpy(y_prob), torch.from_numpy(y_true)).item()
    result = {
        'nll': float(tnll),
        'nll_miss': float(nll_miss),
        'ece': float(ece_val),
        'predictive_entropy': {
            'total': (float(total_entropy.mean()), float(total_entropy.std())),
            'aleatoric': (float(aleatoric.mean()), float(aleatoric.std())),
            'epistemic': (float(epistemic.mean()), float(epistemic.std()))
        },
        **{
            f"top-{k}": float(a) for k, a in enumerate(acc, 1)
        }
    }
    return result


@ ex.automain
def main(_run, num_train_sample, device, validation, num_epochs, logging_freq, kl_type, gamma, entropy_type, det_checkpoint, dataset, save_freq):
    logger=get_logger()
    if validation:
        train_loader, valid_loader, test_loader = get_dataloader()
        logger.info(
            f"Train size: {len(train_loader.dataset)}, validation size: {len(valid_loader.dataset)}, test size: {len(test_loader.dataset)}")
    else:
        train_loader, test_loader= get_dataloader()
        logger.info(
            f"Train size: {len(train_loader.dataset)}, test size: {len(test_loader.dataset)}")
    n_batch= len(train_loader)
    model, optimizer, scheduler= get_model()
    count_parameters(model, logger)
    logger.info(str(model))
    model.train()
    if det_checkpoint != '':
        logger.info(str(model.load_state_dict(
            torch.load(det_checkpoint, device), strict=False)))
    for i in range(num_epochs):
        total_eloglike = 0
        for bx, by in train_loader:
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)
            optimizer.zero_grad()
            eloglike, kl, entropy= model.vi_loss(bx, by, num_train_sample, kl_type, entropy_type)
            viw = get_vi_weight(epoch =i)
            loss= eloglike + viw*(kl - gamma*entropy)/(n_batch*bx.size(0))

            loss.backward()
            optimizer.step()
            total_eloglike += eloglike.detach()
        scheduler.step()
        total_eloglike = total_eloglike.item() / len(train_loader)
        kl = kl.item()
        entropy = entropy.item()
        ex.log_scalar('eloglike.train', total_eloglike, i)
        ex.log_scalar('kl.train', kl, i)
        ex.log_scalar('entropy.train', entropy, i)
        if (i+1) % logging_freq == 0:
            logger.info("VB Epoch %d: loglike: %.4f, kl: %.4f, entropy: %.4f, kl weight: %.4f, lr1: %.4f, lr2: %.4f",
                        i, total_eloglike, kl, entropy, viw, optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'])
        if (i+1) % save_freq == 0:
            torch.save(model.state_dict(), os.path.join(BASE_DIR, _run._id, f'checkpoint{i+1}.pt'))
    torch.save(model.state_dict(), os.path.join(
        BASE_DIR, _run._id, f'checkpoint.pt'))
    logger.info('Save checkpoint')
    model.load_state_dict(torch.load(os.path.join(
        BASE_DIR, _run._id, f'checkpoint.pt'), map_location=device))
    test_result = test_stochastic(model, test_loader)
    os.makedirs(os.path.join(BASE_DIR, _run._id, dataset), exist_ok=True)
    with open(os.path.join(BASE_DIR, _run._id, dataset, 'test_result.json'), 'w') as out:
        json.dump(test_result, out)
    if validation:
        valid_result = test_stochastic(model, valid_loader)
        with open(os.path.join(BASE_DIR, _run._id, dataset, 'valid_result.json'), 'w') as out:
            json.dump(valid_result, out)
    for i in range(5):
        dataloader = get_corruptdataloader(intensity=i)
        result = test_stochastic(model, dataloader)
        os.makedirs(os.path.join(BASE_DIR, _run._id, dataset, str(i)), exist_ok=True)
        with open(os.path.join(BASE_DIR, _run._id, dataset, str(i), 'result.json'), 'w') as out:
            json.dump(result, out)


    
