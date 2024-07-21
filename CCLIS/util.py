from __future__ import print_function

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Sampler, RandomSampler
from scipy.stats import multivariate_normal


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform
    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch):
    lr_enc = args.learning_rate
    lr_prot = args.learning_rate_prototypes
    if args.cosine:
        eta_min_enc = lr_enc * (args.lr_decay_rate ** 3)
        eta_min_prot = lr_prot * (args.lr_decay_rate ** 3)
        lr_enc = eta_min_enc + (lr_enc - eta_min_enc) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
        lr_prot = eta_min_prot + (lr_prot - eta_min_prot) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2        
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr_enc = lr_enc * (args.lr_decay_rate ** steps)
            lr_prot = lr_prot * (args.lr_decay_rate ** steps)

    lr_list = [lr_enc, lr_enc, lr_prot]

    for idx, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr_list[idx]


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr_enc = args.warmup_from_enc + p * (args.warmup_to_enc - args.warmup_from_enc)
        lr_prot = args.warmup_from_prot + p * (args.warmup_to_prot - args.warmup_from_prot)
        lr_list = [lr_enc, lr_enc, lr_prot]

        for idx, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr_list[idx]


def adjust_learning_rate_linear(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2           
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for idx, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr


def warmup_learning_rate_linear(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)
        
        for idx, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr


def set_optimizer(opt, model):
    if 'prototypes.weight' in model.state_dict().keys():
        optimizer = optim.SGD([
                          {'params': model.encoder.parameters()},
                          {'params': model.head.parameters()},
                          {'params': model.prototypes.parameters(), 'lr': opt.learning_rate_prototypes},
                          ],
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    else:
        learning_rate =  opt.learning_rate
        optimizer = optim.SGD(model.parameters(),
                          lr=learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
        
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...'+save_file)
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


def load_model(model, optimizer, save_file):
    print('==> Loading...' + save_file)
    loaded = torch.load(save_file)

    model.load_state_dict(loaded['model'])
    optimizer.load_state_dict(loaded['optimizer'])
    del loaded

    return model, optimizer

class BatchSchedulerSampler(torch.utils.data.sampler.Sampler):
    """
    iterate over tasks and provide a random batch per task in each mini-batch
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset

        self.batch_size = batch_size  # list 
        self.number_of_datasets = len(dataset.datasets) 

        self.largest_dataset_size = max([len(cur_dataset) for cur_dataset in dataset.datasets])
        self.dataset_len = sum([len(cur_dataset) for cur_dataset in self.dataset.datasets])

    def __len__(self):
        return self.dataset_len

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            sampler = RandomSampler(cur_dataset) 
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1] 
        step = sum(self.batch_size) 

        samples_to_grab, epoch_samples = self.batch_size, self.dataset_len  
        # print('epoch_samples', epoch_samples)

        final_samples_list = []  # this is a list of indexes from the combined dataset
        for _ in range(0, epoch_samples, step):
            for i in range(self.number_of_datasets):
                cur_batch_sampler = sampler_iterators[i]
                cur_samples = []
                for _ in range(samples_to_grab[i]):
                    try:
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                    except StopIteration: 
                        # got to the end of iterator - restart the iterator and continue to get samples
                        # until reaching "epoch_samples"
                        break

                final_samples_list.extend(cur_samples)

        return iter(final_samples_list)


