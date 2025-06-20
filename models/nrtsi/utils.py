import os
import pdb
import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

matplotlib.use("Agg")

def nll_gauss(gt, pred, eps=1e-6):
    pred_mean = pred[:, :, : gt.shape[-1]]
    pred_std = F.softplus(pred[:, :, gt.shape[-1] :]) + eps
    normal_distri = torch.distributions.Normal(pred_mean, pred_std)
    LL = normal_distri.log_prob(gt)
    # print(LL.shape) # [bs, time, 44]
    # print(LL.sum(-1).shape) # [bs, time]
    NLL = -LL.sum(-1).mean()
    return NLL

def sample_gauss(pred, gt, gap, eps=1e-6):
    pred_mean = pred[:, :, : gt.shape[-1]]
    pred_std = F.softplus(pred[:, :, gt.shape[-1] :]) + eps
    if gap <= 2**2:
        pred_std = 1e-5 * pred_std
    
    normal_distri = torch.distributions.Normal(pred_mean, pred_std)
    return normal_distri.sample()


def gap_to_max_gap(gap):
    i = 0
    while gap > 2**i:
        i += 1
    return 2**i


def posterior(X_s, X_train, Y_train, len=1.0, sigma_f=1.0, sigma_y=1e-8):
    """
    Computes the suffifient statistics of the posterior distribution
    from m training data X_train and Y_train and n new inputs X_s.

    Args:
        X_s: New input locations (n x d).
        X_train: Training locations (m x d).
        Y_train: Training targets (m x 1).
        len: Kernel length parameter.
        sigma_f: Kernel vertical variation parameter.
        sigma_y: Noise parameter.

    Returns:
        Posterior mean vector (n x d) and covariance matrix (n x n).
    """
    K = kernel(X_train, X_train, len, sigma_f) + sigma_y**2 * np.eye(len(X_train))
    K_s = kernel(X_train, X_s, len, sigma_f)
    K_ss = kernel(X_s, X_s, len, sigma_f) + 1e-8 * np.eye(len(X_s))
    K_inv = np.linalg.inv(K)

    # Equation (7)
    mu_s = K_s.T.dot(K_inv).dot(Y_train)

    # Equation (8)
    cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

    return mu_s, cov_s


def kernel(X1, X2, len=1.0, sigma_f=1.0):
    """
    Isotropic squared exponential kernel.

    Args:
        X1: Array of m points (m x d).
        X2: Array of n points (n x d).

    Returns:
        (m x n) matrix.
    """
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / len**2 * sqdist)


def compute_gp_uncertainty(obs_list, next_list, len=20):
    X_train = np.array(obs_list).reshape(-1, 1)
    Y_train = np.zeros_like(X_train)
    X = np.array(next_list).reshape(-1, 1)
    mu_s, cov_s = posterior(X, X_train, Y_train, len, 1)
    uncertainty = np.diag(np.abs(cov_s))
    min_uncertainty = np.min(uncertainty)
    idx = np.where(uncertainty < 2 * min_uncertainty)[0].tolist()
    return [next_list[each] for each in idx]


def get_next_to_impute(mask, max_level, gp_uncertrain=0):
    mask_ = mask.clone().detach().cpu()
    seq_len = mask_.shape[1]

    obs_list = (mask_[0, :, 0] == 1).nonzero()
    min_dist_to_obs = np.zeros(seq_len)
    for t in range(seq_len):
        if mask_[0, t, 0] == 0:
            min_dist = np.abs((np.array(obs_list) - t)).min()
            if min_dist <= 2**max_level:
                min_dist_to_obs[t] = min_dist
    next_idx = np.argwhere(min_dist_to_obs == np.amax(min_dist_to_obs))[:, 0].tolist()

    gap = np.amax(min_dist_to_obs)
    if gp_uncertrain and gap == 2**max_level:
        next_idx = compute_gp_uncertainty(obs_list, next_idx)
    return next_idx, gap


def get_gap_lr_bs(dataset, epoch, init_lr, use_ta):
    reset_best_loss = False
    save_ckpt = False
    # reset_best_loss_epoch = [100, 200, 300, 400, 500] # basketball
    reset_best_loss_epoch = [700, 1400, 1700, 2700, 3100] # soccer, afootball
    save_ckpt_epoch = [each - 1 for each in reset_best_loss_epoch]
    teacher_forcing = True

    if epoch < reset_best_loss_epoch[0]:
        if epoch < reset_best_loss_epoch[0] - 50:
            min_gap, max_gap, lr, ta = 0, 1, init_lr, teacher_forcing
        else:
            min_gap, max_gap, lr, ta = 0, 1, 0.1 * init_lr, teacher_forcing
    elif epoch < reset_best_loss_epoch[1]:
        if epoch < reset_best_loss_epoch[1] - 50:
            min_gap, max_gap, lr, ta = 1, 2, init_lr, teacher_forcing
        else:
            min_gap, max_gap, lr, ta = 1, 2, 0.1 * init_lr, teacher_forcing
    elif epoch < reset_best_loss_epoch[2]:
        if epoch < reset_best_loss_epoch[2] - 50:
            min_gap, max_gap, lr, ta = 2, 4, init_lr, teacher_forcing
        else:
            min_gap, max_gap, lr, ta = 2, 4, 0.1 * init_lr, teacher_forcing
    elif epoch < reset_best_loss_epoch[3]:
        if epoch < reset_best_loss_epoch[3] - 50:
            min_gap, max_gap, lr, ta = 4, 8, init_lr, teacher_forcing
        else:
            min_gap, max_gap, lr, ta = 4, 8, 0.1 * init_lr, teacher_forcing
    elif epoch < reset_best_loss_epoch[4]:
        if epoch < reset_best_loss_epoch[4] - 50:
            min_gap, max_gap, lr, ta = 8, 16, init_lr, teacher_forcing
        else:
            min_gap, max_gap, lr, ta = 8, 16, 0.1 * init_lr, teacher_forcing
            teacher_forcing = use_ta
    # if epoch < 700:
    #     if epoch < 650:
    #         min_gap, max_gap, lr, ta = 0, 1, init_lr, teacher_forcing
    #     else:
    #         min_gap, max_gap, lr, ta = 0, 1, 0.1 * init_lr, teacher_forcing
    # elif epoch < 1400:
    #     if epoch < 1350:
    #         min_gap, max_gap, lr, ta = 1, 2, init_lr, teacher_forcing
    #     else:
    #         min_gap, max_gap, lr, ta = 1, 2, 0.1 * init_lr, teacher_forcing
    # elif epoch < 1700:
    #     if epoch < 1650:
    #         min_gap, max_gap, lr, ta = 2, 4, init_lr, teacher_forcing
    #     else:
    #         min_gap, max_gap, lr, ta = 2, 4, 0.1 * init_lr, teacher_forcing
    # elif epoch < 2700:
    #     if epoch < 2650:
    #         min_gap, max_gap, lr, ta = 4, 8, init_lr, teacher_forcing
    #     else:
    #         min_gap, max_gap, lr, ta = 4, 8, 0.1 * init_lr, teacher_forcing
    # elif epoch < 3100:
    #     if epoch < 3050:
    #         min_gap, max_gap, lr, ta = 8, 16, init_lr, teacher_forcing
    #     else:
    #         min_gap, max_gap, lr, ta = 8, 16, 0.1 * init_lr, teacher_forcing
    #         teacher_forcing = use_ta

    if epoch in reset_best_loss_epoch:
        reset_best_loss = True
    if epoch in save_ckpt_epoch:
        save_ckpt = True

    train_args = (min_gap, max_gap, lr, ta)

    return train_args, reset_best_loss, save_ckpt
