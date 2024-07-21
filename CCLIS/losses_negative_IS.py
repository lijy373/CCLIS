from __future__ import print_function

import torch
import torch.nn as nn


class ISSupConLoss(nn.Module):
    def __init__(self, temperature=0.07, prototypes_mode='mean',
                 base_temperature=0.07, embedding_shape=512, opt=None):
        super(ISSupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.prototypes_mode = prototypes_mode
        
        self.n_cls = opt.n_cls 
        self.mem_size = opt.mem_size
        self.cls_per_task = opt.cls_per_task
        
        self.embedding_shape = embedding_shape


    def score_calculate(self, output, features, labels=None, importance_weight=None, index=None, target_labels=None, sample_num=[], mask=None, score_mask=[], all_labels=None):
        assert target_labels is not None and len(target_labels) > 0, "Target labels should be given as a list of integer"

        # features: [batch_size, embed_size]
        self.replay_sample_num = torch.tensor(sample_num)
        cur_all_labels = torch.arange(target_labels[-1] + 1)

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu')) 
        
        if len(features.shape) < 2:
            raise ValueError('`features` needs to be [bsz, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 2:
            features = features.view(features.shape[0], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            raise ValueError('`labels` or `mask` should be defined')
        elif labels is not None:
            labels = labels.contiguous()  # labels [batch_size]
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            all_labels = torch.unique(labels).view(-1, 1).to(device)  
            cur_all_labels = cur_all_labels.view(-1, 1).to(device)  
            mask = torch.eq(all_labels, labels.T).float().to(device) 
        else:
            mask = mask.float().to(device)
        
        importance_weight = importance_weight.float().to(device)
        
        if all_labels is not None:
            output = output[:target_labels[-1] + 1, :]
                
        # compute logits
        logits = torch.div(output, self.temperature).to(torch.float64)  # [class_num, batch_size]
        with torch.no_grad():
            cur_class_num = target_labels[-1] + 1
            batch_score_sum = torch.zeros(cur_class_num, cur_class_num)
            score_mat = torch.exp(logits)

            for idx in range(target_labels[-1] + 1):
                batch_score_sum[:, idx] = score_mat[:, labels==idx].sum(1)

        return score_mat, batch_score_sum


    def forward(self, output, features, labels=None, importance_weight=None, index=None, target_labels=None, sample_num=None, mask=None, score_mask=None, all_labels = None):
        assert target_labels is not None and len(target_labels) > 0, "Target labels should be given as a list of integer"

        self.replay_sample_num = torch.tensor(sample_num) if sample_num is not None else torch.Tensor()
        _cur_all_labels = torch.arange(target_labels[-1] + 1)

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu')) 

        if len(features.shape) < 2:
            raise ValueError('`features` needs to be [bsz, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 2:
            features = features.view(features.shape[0], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            raise ValueError('`labels` or `mask` should be defined')
        elif labels is not None:
            labels = labels.contiguous()  # labels [batch_size]
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            all_labels = torch.unique(labels).view(-1, 1).to(device)   
            mask = torch.eq(all_labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        importance_weight = importance_weight.float().to(device)        
        if all_labels is not None:
            prototypes_mask = torch.scatter(
                torch.zeros(len(all_labels), self.n_cls).float().to(device),
                1,
                all_labels.view(-1, 1).to(device),
                1
                )
            output = torch.matmul(prototypes_mask, output)

        # compute logits
        anchor_dot_contrast = torch.div(output, self.temperature).to(torch.float64) # [class_num, batch_size]

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        task_all_labels = all_labels // self.cls_per_task
        task_labels = labels // self.cls_per_task

        if score_mask is not None:
            # print('score_mask', score_mask)
            label_mask = torch.tensor([score_mask[item] for item in labels.tolist()]).to(device)
            score_scale_mask = torch.eq(all_labels, label_mask).float().to(device)  
        else:
            score_scale_mask = torch.ones(len(all_labels), len(labels)).to(device)

        with torch.no_grad():
            _importance_weight = importance_weight * (mask * mask.sum(dim=1, keepdim=True)).sum(dim=0)

        cur_task_mask_col = (task_all_labels != (target_labels[-1] // 2)).float()
        cur_task_mask_row = (task_labels != (target_labels[-1] // 2)).float()

        cur_task_mask = (cur_task_mask_col.view(-1,1) * cur_task_mask_row).to(device)
        all_mask = score_scale_mask * cur_task_mask * (torch.ones_like(mask) - mask)

        _logits = logits - torch.log(_importance_weight) * all_mask
        log_prob = logits - torch.log(torch.exp(_logits).sum(1, keepdim=True))  # normalize

        IS_supcon_loss = - (log_prob * mask).sum() / mask.sum()
        
        return IS_supcon_loss
