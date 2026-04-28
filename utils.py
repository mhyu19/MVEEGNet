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


def plot_metric_history(data_df, title, ylabel, save_path, args):
        plt.figure(figsize=(12, 8))
        colors = plt.get_cmap('tab20', args.subjects)
        for i, col in enumerate(data_df.columns):
            plt.plot(data_df.index + 1, data_df[col], label=col, color=colors(i))
        plt.xlabel('Epochs')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(True)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(save_path)
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

        plt.figure(figsize=(10, 8))
        lw = 2
        plt.plot(fpr["macro"], tpr["macro"], label=f'Macro-average ROC curve (area = {roc_auc["macro"]:.3f})',
                 color='navy', linestyle=':', linewidth=4)
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw, label=f'ROC curve of class {i} (area = {roc_auc[i]:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Aggregated ROC - {plot_title}')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(save_path)
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