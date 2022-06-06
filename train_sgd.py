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
import models
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
    name = ''  # Unique name for the folder of the experiment
    model_name = 'DetWideResNet28x10'  # Choose with model to train
    # the KL weight will increase from <kl_min> to <kl_max> for <last_iter> iterations.
    batch_size = 128  # Batch size
    test_batch_size = 512
    # Universal options for the SGD
    sgd_params = {
        'momentum': 0.9,
        'dampening': 0.0,
        'nesterov': True
    }
    num_epochs = 300  # Number of training epoch
    validation = True  # Whether of not to use a validation set
    validation_fraction = 0.1  # Size of the validation set
    validate_freq = 5  # Frequency of testing on the validation set
    logging_freq = 1  # Logging frequency
    device = 'cuda'
    lr_ratio_det = 0.01  # For annealing the learning rate of the deterministic weights
    # First value chooses which epoch to start decreasing the learning rate and the second value chooses which epoch to stop. See the schedule function for more information.
    det_milestones = (0.5, 0.9)
    augment_data = True
    if not torch.cuda.is_available():
        device = 'cpu'
    dataset = 'cifar100'  # Dataset of the experiment
    if dataset == 'cifar100' or dataset == 'vgg_cifar100':
        num_classes = 100
    elif dataset == 'cifar10' or dataset == 'vgg_cifar10' or dataset == 'fmnist':
        num_classes = 10
    elif dataset == 'tinyimagenet':
        num_classes = 200
    use_sam = False
    sam_params = {
        'rho': 1.0, 'adaptive': True
    }
    bn_momentum = 0.1
    det_checkpoint = ''
    num_train_workers = 4
    num_test_workers = 2


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
def get_model(model_name, num_classes, device, sgd_params, num_epochs, det_milestones, lr_ratio_det, bn_momentum, sam_params, use_sam):
    model = getattr(models, model_name)(num_classes)
    if use_sam:
        base_optimizer = torch.optim.SGD
        optimizer = models.SAM(model.parameters(), base_optimizer, **{**sam_params, **sgd_params})
    else:
        optimizer = torch.optim.SGD(model.parameters(), **sgd_params)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [lambda e: schedule(num_epochs, e, det_milestones, lr_ratio_det)])
    model.to(device)
    return model, optimizer, scheduler


@ex.capture
def get_dataloader(batch_size, test_batch_size, validation, validation_fraction, dataset, augment_data, num_train_workers, num_test_workers):
    return get_data_loader(dataset, train_bs=batch_size, test_bs=test_batch_size, validation=validation, validation_fraction=validation_fraction,
                           augment = augment_data, num_train_workers = num_train_workers, num_test_workers = num_test_workers)

@ex.capture
def get_corruptdataloader(intensity, test_batch_size, dataset, num_test_workers):
    return get_corrupt_data_loader(dataset, intensity, batch_size=test_batch_size, root_dir='data/', num_workers=num_test_workers)

@ex.capture
def get_logger(_run, _log):
    fh=logging.FileHandler(os.path.join(BASE_DIR, _run._id, 'train.log'))
    fh.setLevel(logging.INFO)
    formatter=logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    _log.addHandler(fh)
    return _log


@ex.capture
def test_deterministic(model, dataloader, device, ece_bins):
    tnll = 0
    acc = [0, 0, 0]
    nll_miss = 0
    y_prob = []
    y_true = []
    model.eval()
    with torch.no_grad():
        for bx, by in dataloader:
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)
            prob = model(bx)
            y_prob.append(prob.exp().cpu().numpy())
            y_true.append(by.cpu().numpy())
            top3 = torch.topk(prob, k=3, dim=1, largest=True, sorted=True)[1]
            tnll += torch.nn.functional.nll_loss(prob, by).item() * len(by)
            y_miss = top3[:, 0] != by
            if y_miss.sum().item() > 0:
                prob_miss = prob[y_miss]
                by_miss = by[y_miss]
                nll_miss += torch.nn.functional.nll_loss(
                    prob_miss, by_miss).item() * len(by_miss)
            for k in range(3):
                acc[k] += (top3[:, k] == by).sum().item()
    nll_miss /= len(dataloader.dataset) - acc[0]
    tnll /= len(dataloader.dataset)
    for k in range(3):
        acc[k] /= len(dataloader.dataset)
    acc = np.cumsum(acc)
    y_prob = np.concatenate(y_prob, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    total_entropy = entropy(y_prob, axis=1)
    ece = models.ECELoss(ece_bins)
    ece_val = ece(torch.from_numpy(y_prob), torch.from_numpy(y_true)).item()
    result = {
        'nll': float(tnll),
        'nll_miss': float(nll_miss),
        'ece': float(ece_val),
        'predictive_entropy': {
            'total': (float(total_entropy.mean()), float(total_entropy.std())),
        },
        **{
            f"top-{k}": float(a) for k, a in enumerate(acc, 1)
        }
    }
    return result


@ ex.automain
def main(_run, device, validation, num_epochs, logging_freq, dataset, use_sam):
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
    models.count_parameters(model, logger)
    logger.info(str(model))
    model.train()
    for i in range(num_epochs):
        total_loss = 0
        for bx, by in train_loader:
            if use_sam:
                models.enable_running_stats(model)
            else:
                optimizer.zero_grad()
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)
            pred = model(bx)
            loss = torch.nn.functional.nll_loss(pred, by)
            loss.backward()
            if use_sam:
                optimizer.first_step(zero_grad=True)
                models.disable_running_stats(model)
                pred = model(bx)
                loss = torch.nn.functional.nll_loss(pred, by)
                loss.backward()
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.step()
            total_loss += loss.detach()
        total_loss = total_loss.item() / len(train_loader)
        ex.log_scalar("nll.train", total_loss, i)
        scheduler.step()
        if (i+1) % logging_freq == 0:
            logger.info("Epoch %d: train %.4f, lr %.4f", i, total_loss, optimizer.param_groups[0]['lr'])
    torch.save(model.state_dict(), os.path.join(
        BASE_DIR, _run._id, f'checkpoint.pt'))
    logger.info('Save checkpoint')
    model.load_state_dict(torch.load(os.path.join(
        BASE_DIR, _run._id, f'checkpoint.pt'), map_location=device))
    test_result = test_deterministic(model, test_loader)
    os.makedirs(os.path.join(BASE_DIR, _run._id, dataset), exist_ok=True)
    with open(os.path.join(BASE_DIR, _run._id, dataset, 'test_result.json'), 'w') as out:
        json.dump(test_result, out)
    if validation:
        valid_result = test_deterministic(model, valid_loader)
        with open(os.path.join(BASE_DIR, _run._id, dataset, 'valid_result.json'), 'w') as out:
            json.dump(valid_result, out)
    for i in range(5):
        dataloader = get_corruptdataloader(intensity=i)
        result = test_deterministic(model, dataloader)
        os.makedirs(os.path.join(BASE_DIR, _run._id, dataset, str(i)), exist_ok=True)
        with open(os.path.join(BASE_DIR, _run._id, dataset, str(i), 'result.json'), 'w') as out:
            json.dump(result, out)


    
