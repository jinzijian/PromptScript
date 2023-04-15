import torch
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import random
from .data import samplers

def make_loss(tag, **kwargs):
    if tag == 'SbertContrastiveLoss':
        from sentence_transformers import losses as sbert_losses
        loss_fn = sbert_losses.ContrastiveLoss(kwargs.get('model'))
    elif tag == 'SbertCosineSimilarityLoss':
        from sentence_transformers import losses as sbert_losses
        loss_fn = sbert_losses.CosineSimilarityLoss(kwargs.get('model'))
    elif tag == 'SbertSoftmaxLoss':
        from sentence_transformers import losses as sbert_losses
        loss_fn = sbert_losses.SoftmaxLoss(model=kwargs.get('model'),
                                           sentence_embedding_dimension=kwargs.get('d_emb', 768),
                                           num_labels=kwargs.get('n_label', 2),
                                           concatenation_sent_difference=True,
                                           loss_fct=torch.nn.CrossEntropyLoss(
                                               weight=kwargs.get('weight', None))
                                          )
    elif tag == 'CrossEntropyLoss':
        loss_fn = torch.nn.CrossEntropyLoss(weight=kwargs.get('weight', None))
    else:
        raise ValueError(f'loss={tag} is not supported')
    return loss_fn


def make_optimizer(tag, params, **kwargs):
    if tag == 'SGD':
        optimizer = optim.SGD(params, lr=kwargs.get('lr', 3e-4), momentum=kwargs.get('momentum'),
                              weight_decay=kwargs.get('weight_decay'), nesterov=kwargs.get('nesterov'))
    elif tag == 'Adam':
        optimizer = optim.Adam(params, lr=kwargs.get('lr', 2e-5), betas=kwargs.get('betas'),
                               weight_decay=kwargs.get('weight_decay'))
    elif tag == 'LBFGS':
        optimizer = optim.LBFGS(params, lr=kwargs.get('lr'))
    else:
        raise ValueError(f'optim={tag} is not supported')
    return optimizer


def make_scheduler(tag, optimizer, kwargs):
    if tag == 'None':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[kwargs.get('milestones', 65535)])
    elif tag == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                              step_size=kwargs.get('step_size'), 
                                              gamma=kwargs.get('factor'))
    elif tag == 'MultiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                                                   milestones=kwargs.get('milestones'),
                                                   gamma=kwargs.get('factor'))
    elif tag == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=kwargs.get('gamma', 0.99))
    elif tag == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                         T_max=kwargs.get('num_epochs'), 
                                                         eta_min=kwargs.get('eta_min', 1e-5))
    elif tag == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                         mode=kwargs.get('min'), 
                                                         factor=kwargs.get('factor'),
                                                         patience=kwargs.get('patience'), 
                                                         threshold=kwargs.get('threshold'), 
                                                         threshold_mode='rel',
                                                         min_lr=kwargs.get('min_lr'),
                                                         verbose=False)
    elif tag == 'CyclicLR':
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=kwargs.get('lr'), 
                                                max_lr=10 * kwargs.get('lr'))
    else:
        raise ValueError(f'scheuler={tag} is not supported')
    return scheduler


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_data_loader(dataset, batch_size, shuffle, distributed=False, collate_fn=None, sampler=None, total_samples=None):
    if shuffle:
        if sampler is None:
            if distributed:
                sampler = samplers.DistributedSampler(dataset, total_samples=total_samples)
            # else:
            #     sampler = samplers.DistributedSampler(dataset, rank=0, num_replicas=1, total_samples=total_samples)
        return DataLoader(dataset, batch_size, sampler=sampler, collate_fn=collate_fn)
    else:
        return DataLoader(dataset, batch_size, shuffle=False, collate_fn=collate_fn)