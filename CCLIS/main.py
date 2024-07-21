from __future__ import print_function

import os
import copy
from pkgutil import iter_importers
import sys
import argparse
import shutil
import time
import math
import random
import numpy as np
import gc

import tensorboard_logger as tb_logger
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from torch.utils.data import Subset, Dataset

from torch.utils.data.dataset import ConcatDataset

from datasets import TinyImagenet, IS_Subset
from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model, load_model
from util import BatchSchedulerSampler

from networks.resnet import SupConResNet
from losses_negative_IS import ISSupConLoss


try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--target_task', type=int, default=0)

    parser.add_argument('--resume_target_task', type=int, default=None)

    parser.add_argument('--end_task', type=int, default=None)

    parser.add_argument('--replay_policy', type=str, choices=['random', 'weight'], default='weight')

    parser.add_argument('--mem_size', type=int, default=20)

    parser.add_argument('--cls_per_task', type=int, default=2)

    parser.add_argument('--distill_power', type=float, default=0.6)

    parser.add_argument('--current_temp', type=float, default=0.2,
                        help='temperature for loss function')

    parser.add_argument('--past_temp', type=float, default=0.01,
                        help='temperature for loss function')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=1000,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')
    parser.add_argument('--start_epoch', type=int, default=None)

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate of encoder')
    parser.add_argument('--learning_rate_prototypes', type=float, default=0.01,
                        help='learning rate of prototypes')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'tiny-imagenet', 'cifar100', 'path'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default='~/data/', help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')


    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    parser.add_argument('--replay_sample_num', type=list, default=[], 
                        help='num of replay samples')
    parser.add_argument('--replay_indices_0', type=int, default=-1, 
                        help='num of replay samples')
    parser.add_argument('--freeze_prototypes_niters', type=int, default=5,
                        help="iterations of frozen prototypes")
    parser.add_argument('--distill_type', type=str, default='PRD',
                        help="way to distill the prev knowledge")
    parser.add_argument('--IRD_type', type=str, default="all",
                        help="distilling on all the data or only the past one when use IRD")
    parser.add_argument('--max_iter', type=int, default=5,
                        help='iterations of the score computing')
    parser.add_argument('--seed', type=int, default=0,
                        help='seed for experiments')
    parser.add_argument('--prefix', type=str, default='',
                        help='usage of the file')
    

    opt = parser.parse_args()

    opt.save_freq = opt.epochs // 2


    if opt.dataset == 'cifar10':
        opt.n_cls = 10
        opt.cls_per_task = 2
        opt.size = 32
    elif opt.dataset == 'tiny-imagenet':
        opt.n_cls = 200
        opt.cls_per_task = 20
        opt.size = 64
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
        opt.cls_per_task = 20
        opt.size = 32
    else:
        pass

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = '~/data/'
    opt.save_file = './save_{}_{}_{}'.format(opt.replay_policy, opt.mem_size, opt.prefix)

    opt.model_path = opt.save_file + '/{}_models'.format(opt.dataset)
    opt.tb_path = opt.save_file + '/{}_tensorboard'.format(opt.dataset)
    opt.log_path = opt.save_file + '/logs'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))
    
    # print('separate_sampler', opt.separate_sampler)

    opt.model_name = '{}_{}_{}_lr_{}_{}_decay_{}_bsz_{}_temp_{}_trial_{}_{}_{}_{}_{}_{}_distill_type_{}_freeze_prototypes_niters_{}_seed_{}'.\
        format(opt.dataset, opt.size, opt.model, opt.learning_rate, opt.learning_rate_prototypes,
               opt.weight_decay, opt.batch_size, opt.temp,
               opt.trial,
               opt.start_epoch if opt.start_epoch is not None else opt.epochs, opt.epochs,
               opt.current_temp,
               opt.past_temp,
               opt.distill_power,
               opt.distill_type,
               opt.freeze_prototypes_niters,
               opt.seed
               )

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from_enc = 0.01
        opt.warmup_from_prot = 0.001
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min_encoder = opt.learning_rate * (opt.lr_decay_rate ** 3)
            eta_min_prototypes = opt.learning_rate_prototypes * (opt.lr_decay_rate ** 3)
            opt.warmup_to_enc = eta_min_encoder + (opt.learning_rate - eta_min_encoder) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
            opt.warmup_to_prot = eta_min_prototypes + (opt.learning_rate_prototypes - eta_min_prototypes) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to_enc = opt.learning_rate
            opt.warmup_to_prot = opt.learning_rate_prototypes

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    opt.log_folder = os.path.join(opt.log_path, opt.model_name)
    if not os.path.isdir(opt.log_folder):
        os.makedirs(opt.log_folder)

    return opt


def set_replay_samples(opt, prev_indices=None, prev_importance_weight=None, prev_score=None):

    # is_training = model.training
    # model.eval()

    class IdxDataset(Dataset):
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = indices
        def __len__(self):
            return len(self.dataset)
        def __getitem__(self, idx):
            return self.indices[idx], self.dataset[idx]

    # construct data loader
    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    if opt.dataset == 'cifar10':
        subset_indices = []
        print(opt.data_folder)
        print(os.getcwd())
        val_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=val_transform,
                                         download=True)
        val_targets = np.array(val_dataset.targets)

    elif opt.dataset == 'tiny-imagenet':
        subset_indices = []
        val_dataset = TinyImagenet(root=opt.data_folder,
                                    transform=val_transform,
                                    download=True)
        val_targets = np.array(val_dataset.targets)

    elif opt.dataset == 'cifar100':
        subset_indices = []
        print(opt.data_folder)
        print(os.getcwd())
        val_dataset = datasets.CIFAR100(root=opt.data_folder,
                                         transform=val_transform,
                                         download=True)
        val_targets = np.array(val_dataset.targets)

    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    
    prev_indices_len = 0

    if prev_indices is None:
        prev_indices, prev_importance_weight = [], []
        observed_classes = list(range(0, opt.target_task*opt.cls_per_task))
    else:
        shrink_size = ((opt.target_task - 1) * opt.mem_size / opt.target_task)
        if len(prev_indices) > 0:
            unique_cls = np.unique(val_targets[prev_indices])
            _prev_indices = prev_indices
            prev_indices_len = len(prev_indices)
            prev_indices = []
            prev_weight = prev_importance_weight 
            prev_importance_weight = []

            for c in unique_cls:
                mask = val_targets[_prev_indices] == c

                size_for_c = shrink_size / len(unique_cls)
                p = size_for_c - (shrink_size // len(unique_cls))  

                if random.random() < p:
                    size_for_c = math.ceil(size_for_c)
                else:
                    size_for_c = math.floor(size_for_c)

                store_index = torch.multinomial(torch.tensor(prev_score[:prev_indices_len])[mask], min(len(torch.tensor(prev_score[:prev_indices_len])[mask]), size_for_c), replacement=False)  # score tensor [old_samples_num] 

                prev_indices += torch.tensor(_prev_indices)[mask][store_index].tolist()

                prev_cur_weight = torch.tensor(prev_score[:prev_indices_len])[mask]

                prev_importance_weight += (prev_cur_weight / prev_cur_weight.sum())[store_index].tolist()

            print(np.unique(val_targets[prev_indices], return_counts=True))
        observed_classes = list(range(max(opt.target_task-1, 0)*opt.cls_per_task, (opt.target_task)*opt.cls_per_task))

    if len(observed_classes) == 0:
        return prev_indices, prev_importance_weight, val_targets
    
    observed_indices = []
    for tc in observed_classes:
        observed_indices += np.where(val_targets == tc)[0].tolist()

    val_observed_targets = val_targets[observed_indices]
    val_unique_cls = np.unique(val_observed_targets)


    selected_observed_indices = []
    selected_observed_importance_weight = []
    for c_idx, c in enumerate(val_unique_cls):
        size_for_c_float = ((opt.mem_size - len(prev_indices) - len(selected_observed_indices)) / (len(val_unique_cls) - c_idx))
        p = size_for_c_float -  ((opt.mem_size - len(prev_indices) - len(selected_observed_indices)) // (len(val_unique_cls) - c_idx))
        if random.random() < p:
            size_for_c = math.ceil(size_for_c_float)
        else:
            size_for_c = math.floor(size_for_c_float)

        mask = val_targets[observed_indices] == c
        store_index = torch.multinomial(torch.tensor(prev_score[prev_indices_len:])[mask], size_for_c, replacement=False)

        selected_observed_indices += torch.tensor(observed_indices)[mask][store_index].tolist()

        observed_cur_weight = torch.tensor(prev_score[prev_indices_len:])[mask] 
        observed_normalized_weight = observed_cur_weight / observed_cur_weight.sum() 

        selected_observed_importance_weight += observed_normalized_weight[store_index].tolist()  

    print(np.unique(val_targets[selected_observed_indices], return_counts=True))
    print(selected_observed_importance_weight)

    return prev_indices + selected_observed_indices, prev_importance_weight + selected_observed_importance_weight, val_targets


def set_loader(opt, replay_indices, importance_weight, training=True):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'tiny-imagenet':
        mean = (0.4802, 0.4480, 0.3975)
        std = (0.2770, 0.2691, 0.2821)
    elif opt.dataset == 'cifar100':
        mean = (0.5153, 0.4961, 0.4497)
        std = (0.2608, 0.2551, 0.2779)
    elif opt.dataset == 'path':
        mean = eval(opt.mean)
        std = eval(opt.mean)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))


    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.RandomResizedCrop(size=opt.size, scale=(0.1 if opt.dataset=='tiny-imagenet' else 0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=opt.size//20*2+1, sigma=(0.1, 2.0))], p=0.5 if opt.size>32 else 0.0),
        transforms.ToTensor(),
        normalize,
    ])


    target_classes = list(range(opt.target_task*opt.cls_per_task, (opt.target_task+1)*opt.cls_per_task))
    print('target_classes', target_classes)

    # separate_sampler = True

    if opt.dataset == 'cifar10':
        subset_indices = []
        subset_importance_weight = []

        _train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=train_transform,
                                         download=True)
        for tc in target_classes:
            target_class_indices = np.where(np.array(_train_dataset.targets) == tc)[0]
            subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()  # cur_sample index, list
            tc_num = (np.array(_train_dataset.targets) == tc).sum()
            
            subset_importance_weight += list(np.ones(tc_num) / tc_num)  # cur_sample importance weight, list

        _subset_indices, _subset_importance_weight = copy.deepcopy(subset_indices), copy.deepcopy(subset_importance_weight)

        if len(replay_indices) > 0 and training:
            prev_dataset = IS_Subset(_train_dataset, replay_indices, importance_weight)
            cur_dataset = IS_Subset(_train_dataset, _subset_indices, _subset_importance_weight)

            dataset_len_list = [len(prev_dataset), len(cur_dataset)]

            train_dataset = ConcatDataset([prev_dataset, cur_dataset])

        else:
            _subset_indices += replay_indices
            _subset_importance_weight += importance_weight

            train_dataset = IS_Subset(_train_dataset, _subset_indices, _subset_importance_weight)

        subset_indices += replay_indices
        subset_importance_weight += importance_weight

        uk, uc = np.unique(np.array(_train_dataset.targets)[subset_indices], return_counts=True)  
        print('uc[np.argsort(uk)]', uc[np.argsort(uk)])
        replay_sample_num = uc[np.argsort(uk)]

    elif opt.dataset == 'tiny-imagenet':
        subset_indices = []
        subset_importance_weight = []
        _train_dataset = TinyImagenet(root=opt.data_folder,
                                          transform=train_transform,
                                          download=True)
        for tc in target_classes:
            target_class_indices = np.where(_train_dataset.targets == tc)[0]
            subset_indices += np.where(_train_dataset.targets == tc)[0].tolist()
            tc_num = (np.array(_train_dataset.targets) == tc).sum()
            
            subset_importance_weight += list(np.ones(tc_num) / tc_num)  # cur_sample importance weight, list

        _subset_indices, _subset_importance_weight = copy.deepcopy(subset_indices), copy.deepcopy(subset_importance_weight)

        if len(replay_indices) > 0 and training:
            prev_dataset = IS_Subset(_train_dataset, replay_indices, importance_weight)
            cur_dataset = IS_Subset(_train_dataset, _subset_indices, _subset_importance_weight)

            dataset_len_list = [len(prev_dataset), len(cur_dataset)]

            train_dataset = ConcatDataset([prev_dataset, cur_dataset])

        else:
            _subset_indices += replay_indices
            _subset_importance_weight += importance_weight
            print('_subset_indices length', len(_subset_indices))
            train_dataset = IS_Subset(_train_dataset, _subset_indices, _subset_importance_weight)

        subset_indices += replay_indices
        subset_importance_weight += importance_weight

        print('dataset length', len(_train_dataset), len(train_dataset))
        print('Dataset size: {}'.format(len(subset_indices)))

        uk, uc = np.unique(np.array(_train_dataset.targets)[subset_indices], return_counts=True)  
        print('uc[np.argsort(uk)]', uc[np.argsort(uk)])
        replay_sample_num = uc[np.argsort(uk)]

    elif opt.dataset == 'cifar100':
        subset_indices = []
        subset_importance_weight = []

        _train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                         transform=train_transform,
                                         download=True)
        for tc in target_classes:
            target_class_indices = np.where(np.array(_train_dataset.targets) == tc)[0]
            subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()  # cur_sample index, list
            tc_num = (np.array(_train_dataset.targets) == tc).sum()

            subset_importance_weight += list(np.ones(tc_num) / tc_num)  # cur_sample importance weight, list
        
        _subset_indices, _subset_importance_weight = copy.deepcopy(subset_indices), copy.deepcopy(subset_importance_weight)

        if len(replay_indices) > 0 and training:
            prev_dataset = IS_Subset(_train_dataset, replay_indices, importance_weight)
            cur_dataset = IS_Subset(_train_dataset, _subset_indices, _subset_importance_weight)

            dataset_len_list = [len(prev_dataset), len(cur_dataset)]

            train_dataset = ConcatDataset([prev_dataset, cur_dataset])

        else:
            _subset_indices += replay_indices
            _subset_importance_weight += importance_weight
            print('_subset_indices length', len(_subset_indices))
            train_dataset = IS_Subset(_train_dataset, _subset_indices, _subset_importance_weight)

        subset_indices += replay_indices
        subset_importance_weight += importance_weight

        print('dataset length', len(_train_dataset), len(train_dataset))        
        print('Dataset size: {}'.format(len(subset_indices)))

        uk, uc = np.unique(np.array(_train_dataset.targets)[subset_indices], return_counts=True)  
        print('uc[np.argsort(uk)]', uc[np.argsort(uk)])
        replay_sample_num = uc[np.argsort(uk)]

    elif opt.dataset == 'path':
        train_dataset = datasets.ImageFolder(root=opt.data_folder,
                                            transform=train_transform)
    else:
        raise ValueError(opt.dataset)

    if len(replay_indices) > 0 and training: 
        train_batch_size_list = [int(np.round(opt.batch_size * dataset_len_list[0] / sum(dataset_len_list))), 
                                 opt.batch_size - int(np.round(opt.batch_size * dataset_len_list[0] / sum(dataset_len_list)))]
        
        print('train_batch_size', train_batch_size_list)
        train_sampler = BatchSchedulerSampler(dataset=train_dataset, batch_size=train_batch_size_list)
        print('len_data', [len(cur_dataset) for cur_dataset in train_sampler.dataset.datasets])
    else:
        train_sampler = None
        
    if training:
        train_loader = torch.utils.data.DataLoader(
                            train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
                            num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    else:
        train_loader = torch.utils.data.DataLoader(
                            train_dataset, batch_size=opt.batch_size, shuffle=False,
                            num_workers=opt.num_workers, pin_memory=True)
        print('no separate sampler')
    
    return train_loader, subset_indices, replay_sample_num 


def set_model(opt):
    # model = SupConResNet(name=opt.model, opt=opt)
    model = SupConResNet(name=opt.model, opt=opt)
    criterion = ISSupConLoss(temperature=opt.temp, opt=opt)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, model2, criterion, optimizer, epoch, subset_sample_num, score_mask, opt):


    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    distill = AverageMeter()

    end = time.time()

    index_list, score = [], []
    distill_type = opt.distill_type
    IRD_type = opt.IRD_type


    for idx, (images, labels, importance_weight, index) in enumerate(train_loader):

        data_time.update(time.time() - end)
        index_list += index
        
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # normalize the prototypes
        with torch.no_grad():
            prev_task_mask = labels < opt.target_task * opt.cls_per_task

            w = model.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            model.prototypes.weight.copy_(w)

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        
        features, output = model(images)

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
            
        target_labels = list(range(opt.target_task*opt.cls_per_task, (opt.target_task+1)*opt.cls_per_task))

        # ISSupCon
        loss = criterion(output,
                         features, 
                         labels, 
                         importance_weight, 
                         index, 
                         target_labels=target_labels, 
                         sample_num=subset_sample_num, 
                         score_mask=score_mask)
        
        if distill_type == 'IRD':
            if opt.target_task > 0:
                # IRD (cur)
                labels_mask = labels < min(target_labels)

                features1_prev_task = features[labels_mask] if IRD_type == 'prev' else features

                features1_sim = torch.div(torch.matmul(features1_prev_task, features1_prev_task.T), opt.current_temp)
                logits_mask = torch.scatter(
                    torch.ones_like(features1_sim),
                    1,
                    torch.arange(features1_sim.size(0)).view(-1, 1).cuda(non_blocking=True),
                    0
                )
                logits_max1, _ = torch.max(features1_sim * logits_mask, dim=1, keepdim=True)
                features1_sim = features1_sim - logits_max1.detach()
                row_size = features1_sim.size(0)
                logits1 = torch.exp(features1_sim[logits_mask.bool()].view(row_size, -1)) / torch.exp(features1_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)

                # IRD (past)
                with torch.no_grad():
                    features2, _ = model2(images)
                    features2_prev_task = features2[labels_mask] if IRD_type == 'prev' else features2

                    features2_sim = torch.div(torch.matmul(features2_prev_task, features2_prev_task.T), opt.past_temp)
                    logits_max2, _ = torch.max(features2_sim*logits_mask, dim=1, keepdim=True)
                    features2_sim = features2_sim - logits_max2.detach()
                    logits2 = torch.exp(features2_sim[logits_mask.bool()].view(row_size, -1)) /  torch.exp(features2_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)

                loss_distill = (-logits2 * torch.log(logits1)).sum(1).mean()
                loss += opt.distill_power * loss_distill
                distill.update(loss_distill.item(), bsz)
        elif distill_type == 'PRD':
            if opt.target_task > 0:
                all_labels = torch.unique(labels).view(-1, 1)

                prev_all_labels = torch.arange(target_labels[0])
                
                prototypes_mask = torch.scatter(
                    torch.zeros(len(prev_all_labels), opt.n_cls).float(),
                    1,
                    prev_all_labels.view(-1,1),
                    1
                    ).to(device)

                labels_mask = labels < min(target_labels)

                # PRD (cur)
                sim_prev_task = torch.matmul(prototypes_mask, output)

                features1_sim = torch.div(sim_prev_task, opt.current_temp)
                 

                logits_max1, _ = torch.max(features1_sim, dim=0, keepdim=True)
                features1_sim = features1_sim - logits_max1.detach()  # number stability
                row_size = features1_sim.size(0)
                
                logits1 = torch.exp(features1_sim) / torch.exp(features1_sim).sum(dim=0, keepdim=True)

                # PRD (past)
                with torch.no_grad():
                    _, sim2_prev_task = model2(images)
                    sim2_prev_task = torch.matmul(prototypes_mask, sim2_prev_task)

                    features2_sim = torch.div(sim2_prev_task, opt.past_temp)
                    logits_max2, _ = torch.max(features2_sim, dim=0, keepdim=True)
                    features2_sim = features2_sim - logits_max2.detach()
                    logits2 = torch.exp(features2_sim) /  torch.exp(features2_sim).sum(dim=0, keepdim=True)

                loss_distill = (-logits2 * torch.log(logits1)).sum(0).mean()
                loss += opt.distill_power * loss_distill
                distill.update(loss_distill.item(), bsz)
        else:
            raise ValueError("distill type {} is not supported".format(distill_type))
        
        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0 or idx+1 == len(train_loader):
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f} {distill.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, distill=distill))
            sys.stdout.flush()

    return losses.avg, model2


def score_computing(val_loader, model, model2, criterion, subset_sample_num, score_mask, opt):
    
    model.eval()
    max_iter = opt.max_iter
    
    for k, v in model.named_parameters():
        if k == 'prototypes.weight':
            print(k, v)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    distill = AverageMeter()

    cur_task_n_cls = (opt.target_task + 1)*opt.cls_per_task
    len_val_loader = sum(subset_sample_num)
    print('val_loader length', len_val_loader)

    end = time.time()

    all_score_sum = torch.zeros(cur_task_n_cls, cur_task_n_cls)
    _score = torch.zeros(cur_task_n_cls, len_val_loader)

    for i in range(max_iter):

        index_list, score_list, label_list = [], [], []
        score_sum = torch.zeros(cur_task_n_cls, cur_task_n_cls)

        for idx, (images, labels, importance_weight, index) in enumerate(val_loader):
            index_list += index
            label_list += labels
        
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]

            with torch.no_grad():
                prev_task_mask = labels < opt.target_task * opt.cls_per_task
        
                features, output = model(images)

                # ISSupCon
                score_mat, batch_score_sum  = criterion.score_calculate(output, features, labels, importance_weight, index, target_labels=list(range(opt.target_task*opt.cls_per_task, (opt.target_task+1)*opt.cls_per_task)), sample_num = subset_sample_num, score_mask=score_mask)
                score_list.append(score_mat)

                score_sum += batch_score_sum

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        index_list = torch.tensor(index_list) 
        label_list = torch.tensor(label_list).tolist()  

        mask = torch.eye(cur_task_n_cls)
        label_score_mask = torch.eq(torch.arange(cur_task_n_cls).view(-1, 1), torch.tensor(label_list)) 

        _score_list = torch.concat(score_list, dim=1) 
        _score_list = _score_list.to('cpu')

        _score -= _score * label_score_mask
        _score += (_score_list / _score_list.sum(dim=1, keepdim=True)) 
        all_score_sum += score_sum 
        all_score_sum -= all_score_sum * mask

    _score /= max_iter
    all_score_sum /= max_iter

    score_class_mask = None
    score = _score.cpu().sum(dim=0) / (_score.shape[0] - 1)

        
    return score_class_mask, index_list, score, model2

def main():
    opt = parse_option()

    target_task = opt.target_task

    seed = opt.seed

    torch.manual_seed(seed)            
    torch.cuda.manual_seed(seed)       
    torch.cuda.manual_seed_all(seed)   

    random.seed(seed)
    np.random.seed(seed)

    # build model and criterion
    model, criterion = set_model(opt)
    
    model2, _ = set_model(opt)
    model2.eval()

    # build optimizer
    optimizer = set_optimizer(opt, model)

    replay_indices, importance_weight, score, score_mask = None, None, None, None
    # print('score_mask', score_mask)

    if opt.resume_target_task is not None:
        load_file = os.path.join(opt.save_folder, 'last_{policy}_{target_task}.pth'.format(policy=opt.replay_policy ,target_task=opt.resume_target_task))
        model, optimizer = load_model(model, optimizer, load_file)
        if opt.resume_target_task == 0:
            replay_indices, importance_weight = [], []
        else:
            replay_indices = np.load(
              os.path.join(opt.log_folder, 'replay_indices_{policy}_{target_task}.npy'.format(policy=opt.replay_policy ,target_task=opt.resume_target_task))
            ).tolist()
            importance_weight = np.load(
              os.path.join(opt.log_folder, 'importance_weight_{policy}_{target_task}.npy'.format(policy=opt.replay_policy ,target_task=opt.resume_target_task))
            ).tolist()
        score = np.load(
            os.path.join(opt.log_folder, 'score_{policy}_{target_task}.npy'.format(policy=opt.replay_policy ,target_task=opt.resume_target_task))
        ).tolist()
        print(len(replay_indices), len(importance_weight), len(score))

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    original_epochs = opt.epochs

    if opt.end_task is not None:
        if opt.resume_target_task is not None:
            assert opt.end_task > opt.resume_target_task
        opt.end_task = min(opt.end_task+1, opt.n_cls // opt.cls_per_task)
    else:
        opt.end_task = opt.n_cls // opt.cls_per_task

    for target_task in range(0 if opt.resume_target_task is None else opt.resume_target_task+1, opt.end_task):

        opt.target_task = target_task
        model2 = copy.deepcopy(model)

        print('Start Training current task {}'.format(opt.target_task))
                
        # acquire replay sample indices
        replay_indices, importance_weight, val_targets = set_replay_samples(opt, prev_indices=replay_indices, prev_importance_weight=importance_weight, prev_score=score)  # [prev_sample_num] tensor

        print('replay_indices', replay_indices)
        if target_task != 0:
            opt.replay_indices_0 = replay_indices[0]

        np.save(
          os.path.join(opt.log_folder, 'replay_indices_{policy}_{target_task}.npy'.format(policy=opt.replay_policy ,target_task=target_task)),
          np.array(replay_indices))
        np.save(
          os.path.join(opt.log_folder, 'importance_weight_{policy}_{target_task}.npy'.format(policy=opt.replay_policy ,target_task=target_task)),
          np.array(importance_weight))

        # build data loader (dynamic: 0109)
        train_loader, subset_indices, subset_sample_num = set_loader(opt, replay_indices, importance_weight)

        np.save(
          os.path.join(opt.log_folder, 'subset_indices_{policy}_{target_task}.npy'.format(policy=opt.replay_policy ,target_task=target_task)),
          np.array(subset_indices))


        # training routine
        if target_task == 0 and opt.start_epoch is not None:
            opt.epochs = opt.start_epoch
        else:
            opt.epochs = original_epochs

        for epoch in range(1, opt.epochs + 1):

            adjust_learning_rate(opt, optimizer, epoch)

            # train for one epoch
            time1 = time.time()
            # print('train_score_mask', score_mask)
            loss, model2 = train(train_loader, model, model2, criterion, optimizer, epoch, subset_sample_num, score_mask, opt)
            time2 = time.time()
            print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

            # tensorboard logger
            logger.log_value('loss_{target_task}'.format(target_task=target_task), loss, epoch)
            logger.log_value('learning_rate_{target_task}_encoder'.format(target_task=target_task), optimizer.param_groups[0]['lr'], epoch)
            logger.log_value('learning_rate_{target_task}_prototypes'.format(target_task=target_task), optimizer.param_groups[2]['lr'], epoch)

        print('val_replay_indices', replay_indices)
        print('val_importance_weight', importance_weight)
        # construct val_loader without data shuffle
        val_loader, _, _ = set_loader(opt, replay_indices, importance_weight, training=False)  

        print('score_mask', score_mask)
        score_mask, index, _score, model2 = score_computing(val_loader, model, model2, criterion, subset_sample_num, score_mask, opt)  # compute score 
        

        print(opt.target_task)
        observed_classes = list(range(opt.target_task * opt.cls_per_task, (opt.target_task + 1) * opt.cls_per_task))

        observed_indices = []
        for tc in observed_classes:
            observed_indices += np.where(val_targets == tc)[0].tolist()

        print('replay_indices_len', len(replay_indices))
        print('observed_indices_len', len(observed_indices))
        score_indices = replay_indices + observed_indices

        score_dict = dict(zip(np.array(index), _score))
        score = torch.stack([score_dict[key] for key in score_indices])
        print('score', score)
        
        # save the last score
        np.save(
          os.path.join(opt.log_folder, 'score_{policy}_{target_task}.npy'.format(policy=opt.replay_policy ,target_task=target_task)),
          np.array(score.cpu()))

        # save the last model
        save_file = os.path.join(
            opt.save_folder, 'last_{policy}_{target_task}.pth'.format(policy=opt.replay_policy ,target_task=target_task))
        save_model(model, optimizer, opt, opt.epochs, save_file)

if __name__ == '__main__':
    main()

    