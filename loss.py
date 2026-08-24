import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import torch
# MK_MMD
def mmd_rbf(source, target, source_labels=None, target_labels=None, kernel_mul=2.0, kernel_num=5, max_samples=1000,
            fix_sigma=None):
    """
    改进版自适应多核 MMD，支持类条件对齐和动态带宽调整

    参数:
    - source/target: 源域/目标域特征 (batch_size, feature_dim)
    - source_labels/target_labels: 类标签 (batch_size,)
    - kernel_mul: 核带宽倍数
    - kernel_num: 核数量
    - max_samples: 中位数估计最大采样数
    - fix_sigma: 固定带宽 (None时自动计算)

    返回:
    - mmd_loss: 加权后的 MMD 损失
    """
    # 类条件对齐模式
    if source_labels is not None and target_labels is not None:
        return _class_conditional_mmd(
            source, target, source_labels, target_labels,
            kernel_mul, kernel_num, max_samples, fix_sigma
        )

    # 计算基础核矩阵
    xx_kernel = _compute_kernel(source, source,
                                kernel_mul, kernel_num,
                                max_samples, fix_sigma)
    yy_kernel = _compute_kernel(target, target,
                                kernel_mul, kernel_num,
                                max_samples, fix_sigma)
    xy_kernel = _compute_kernel(source, target,
                                kernel_mul, kernel_num,
                                max_samples, fix_sigma)

    # 计算 MMD
    mmd = (xx_kernel.mean() + yy_kernel.mean() - 2 * xy_kernel.mean())
    return abs(mmd)


def _class_conditional_mmd(source, target, source_labels, target_labels,
                           kernel_mul, kernel_num, max_samples, fix_sigma):
    """类条件 MMD 计算"""
    classes = torch.unique(torch.cat([source_labels, target_labels]))
    total_mmd = 0.0
    valid_classes = 0

    for cls in classes:
        src_mask = (source_labels == cls)
        tgt_mask = (target_labels == cls)

        if src_mask.sum() == 0 or tgt_mask.sum() == 0:
            continue

        # 计算单类 MMD
        cls_mmd = mmd_rbf(source[src_mask], target[tgt_mask],
                          None, None, kernel_mul,
                          kernel_num, max_samples, fix_sigma)
        total_mmd += cls_mmd
        valid_classes += 1

    return total_mmd / valid_classes if valid_classes > 0 else 0.0


def _compute_kernel(x, y, kernel_mul, kernel_num, max_samples, fix_sigma):
    """自适应带宽核矩阵计算"""
    n_x, n_y = x.size(0), y.size(0)

    # 动态带宽估计
    if fix_sigma is None:
        # 随机采样估计中位数
        if max_samples < n_x * n_y:
            indices = torch.randperm(n_x * n_y)[:max_samples]
            x_flat = x.view(-1, x.shape[-1])
            y_flat = y.view(-1, y.shape[-1])
            sampled_pairs = x_flat[indices // n_y] - y_flat[indices % n_y]
            sigma = torch.median(torch.norm(sampled_pairs, dim=1) ** 2)
        else:
            sigma = torch.median(torch.cdist(x, y) ** 2)

        base_sigma = 2.0 * sigma  # 经验缩放因子
        sigma_list = [base_sigma * (kernel_mul ** i) for i in range(kernel_num)]
    else:
        sigma_list = [fix_sigma * (kernel_mul ** i) for i in range(kernel_num)]

    # 数值稳定性处理
    sigma_list = [max(s, 1e-6) for s in sigma_list]

    # 向量化核计算
    x = x.view(n_x, 1, -1)
    y = y.view(1, n_y, -1)
    l2_distance = torch.sum((x - y) ** 2, dim=2)  # (n_x, n_y)

    kernel_val = 0.0
    for sigma in sigma_list:
        kernel_val += torch.exp(-l2_distance / (2 * sigma))

    return kernel_val

def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device, useKL=True):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    if not useKL:
        return A

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return A + kl_div


def edl_digamma_loss(alpha, target, epoch_num, num_classes, annealing_step, device):
    loss = edl_loss(torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, device)
    return torch.mean(loss)


def get_dc_loss(evidences, device):
    num_views = len(evidences)
    batch_size, num_classes = evidences[0].shape[0], evidences[0].shape[1]
    p = torch.zeros((num_views, batch_size, num_classes)).to(device)
    u = torch.zeros((num_views, batch_size)).to(device)
    for v in range(num_views):
        alpha = evidences[v] + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        p[v] = alpha / S
        u[v] = torch.squeeze(num_classes / S)
    dc_sum = 0
    for i in range(num_views):
        pd = torch.sum(torch.abs(p - p[i]) / 2, dim=2) / (num_views - 1)  # (num_views, batch_size)
        cc = (1 - u[i]) * (1 - u)  # (num_views, batch_size)
        dc = pd * cc
        dc_sum = dc_sum + torch.sum(dc, dim=0)
    dc_sum = torch.mean(dc_sum)
    return dc_sum


def get_loss(evidences, evidence_a, target, epoch_num, num_classes, annealing_step, gamma, device):
    target = F.one_hot(target, num_classes)
    alpha_a = evidence_a + 1
    loss_acc = edl_digamma_loss(alpha_a, target, epoch_num, num_classes, annealing_step, device)
    for v in range(len(evidences)):
        alpha = evidences[v] + 1
        loss_acc += edl_digamma_loss(alpha, target, epoch_num, num_classes, annealing_step, device)
    loss_acc = loss_acc / (len(evidences) + 1)
    loss = loss_acc + gamma * get_dc_loss(evidences, device)
    return loss


def kl_divergence(alpha, num_classes, device):
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl


def kl_divergence(alpha, num_classes, device):
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl

def ce_loss(p, alpha, c, global_step, annealing_step, device):
    """
    计算论文中使用的基于证据学习的损失函数，结合了分类准确性和不确定性。
    这个损失函数旨在鼓励模型对正确类别的证据更强，并抑制错误类别的证据。

    Args:
        p (torch.Tensor): 形状为 (batch_size,) 的张量，包含每个样本的真实标签索引。
        alpha (torch.Tensor): 形状为 (batch_size, n_classes) 的张量，代表狄利克雷分布的参数。
        c (int): 类别总数。
        global_step (int): 当前训练步数。
        annealing_step (int): 退火步数，用于控制 KL 项的权重。
        device (torch.device): 当前使用的设备。

    Returns:
        torch.Tensor: 形状为 () 的标量张量，表示平均损失值。
    """
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    B = annealing_coef * kl_divergence(alp, c, device)
    return torch.mean((A + B))
