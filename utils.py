import os
import logging
from termcolor import colored
import time
import numpy as np
import random
import torch
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt 
from itertools import cycle
from torch_geometric.data import Data, Batch



def set_random_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed) 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  


def create_logger(dir_path, data_name, model_name, log_file):
    os.makedirs(dir_path, exist_ok=True)
    time_str = time.strftime('%m-%d-%H-%M')
    log_file = '{}_{}_{}_{}.log'.format(model_name, data_name, log_file, time_str)
    final_log_file = os.path.join(dir_path, log_file)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    #
    fmt = '[%(asctime)s] %(message)s'
    color_fmt = colored('[%(asctime)s]', 'green') + ' %(message)s'

    file = logging.FileHandler(filename=final_log_file, mode='a')
    file.setLevel(logging.INFO)
    file.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(console)

    return logger


def plot_metric_history(data_df, title, ylabel, save_path, subject):
    plt.figure(figsize=(10, 8))
    colors = plt.get_cmap('tab20', subject)
    for i, col in enumerate(data_df.columns):
        plt.plot(data_df.index + 1, data_df[col], color=colors(i), label=col)
        
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16)
    
    # 绘制图例并放置在图表右侧外
    plt.legend(loc='lower right', fontsize=10) # 考虑到15个受试者，字体稍微调小为10
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 确保保存路径的文件夹存在
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()
    print(f"Plot saved to {save_path}")


def plot_roc_curves(y_true, y_score, plot_title, save_path):
    n_classes = y_true.shape[1]
    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # 稍微调整了 figsize 比例 (8, 6) 视觉上更紧凑
    plt.figure(figsize=(8, 6))
    lw = 2
    plt.plot(fpr["macro"], tpr["macro"], label=f'Macro-average ROC curve (area = {roc_auc["macro"]:.3f})',
             color='navy', linestyle=':', linewidth=4)
             
    # 修改了颜色列表，使用 tab10 色板，完美支持高达 10 个类别不重复
    colors = plt.get_cmap('tab10').colors
    for i, color in zip(range(n_classes), cycle(colors)):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw, label=f'ROC curve of class {i} (area = {roc_auc[i]:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'Aggregated ROC - {plot_title}', fontsize=14)
    
    # 缩小图例字体以防遮挡曲线
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # ================= 核心修改区域 =================
    # 1. 自动调整子图参数，使之填充整个图像区域
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    # 2. 加入 bbox_inches='tight'，在保存时彻底裁切掉多余的白色外边距
    # pad_inches=0.02 仅保留极小的一圈缓冲，防止刻度数字被切除
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
    # ===============================================
    
    plt.close()
    print(f"Plot saved to {save_path}")

def add_gaussian_noise(source_data, target_view_idx, noise_level):
    """
    向指定视图注入高斯噪声
    """
    # 五个视图，每个视图的数据存储在 source_data 中的不同键下，如 'view_0', 'view_1', ..., 'view_4'
    noisy_data = source_data.clone()
    # 2. 构造目标视图的键名 (例如 'view_3')
    target_view_name = f'view_{target_view_idx}'
    # 3. 检查该视图是否存在
    if target_view_name in noisy_data.node_types:
        # 获取原始特征矩阵 [930, 265]
        x = noisy_data[target_view_name].x
        # 4. 生成与特征矩阵形状一致的高斯噪声
        # 使用 torch.randn_like 确保设备(CPU/GPU)和数据类型一致
        noise = torch.randn_like(x) * noise_level
        # 5. 叠加噪声并重新赋值
        noisy_data[target_view_name].x = x + noise
        
    return noisy_data


# 定义调度器类，用于动态调整训练参数
class ProgressiveDomainScheduler:
    def __init__(self, total_epochs, max_mix_domains=3):
        self.total_epochs = total_epochs  # 总训练轮数
        self.max_mix_domains = max_mix_domains  # 最大混合域数量（这里为3）

    def get_params(self, epoch):
        progress = epoch / self.total_epochs
        mix_ratio = min(1.0, progress)
        mmd_weight = 0.1
        domain_weight = 0.1
        return {'mix_ratio': mix_ratio, 'mmd_weight': mmd_weight, 'domain_weight': domain_weight}
