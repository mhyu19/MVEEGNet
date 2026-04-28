import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import DenseSAGEConv
from torch.nn import Linear, Dropout
from torch_geometric.utils import to_dense_batch
from evidence import FML


class IterativeGraphLearner(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=4, topk=10, epsilon=0.5):
        super().__init__()
        self.k = topk
        self.epsilon = epsilon
        self.num_heads = num_heads
        
        # 多头注意力权重
        self.weight_tensor = nn.Parameter(torch.Tensor(num_heads, hidden_dim))
        nn.init.xavier_uniform_(self.weight_tensor)
        
        # 非线性变换
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        """
        x: (Batch, N, Dim)
        返回: adj (Batch, N, N), smoothness_loss
        """
        B, N, D = x.shape
        # 变换特征
        x_trans = self.activation(self.linear(x)) # (B, N, Hidden)
        adj_list = []
        for i in range(self.num_heads):
            w = self.weight_tensor[i].unsqueeze(0).unsqueeze(0) # (1, 1, Hidden)
            x_weighted = x_trans * w
            # 计算余弦相似度
            x_norm = F.normalize(x_weighted, p=2, dim=-1)
            adj = torch.bmm(x_norm, x_norm.transpose(1, 2)) # (B, N, N)
            adj_list.append(adj)
        # 平均多头结果
        adj = torch.mean(torch.stack(adj_list), dim=0)
        # 稀疏化
        # 方式 A: 阈值截断 (Thresholding) -> 去除噪声边
        mask_threshold = (adj > self.epsilon).float()
        # 方式 B: Top-K
        topk_values, topk_indices = torch.topk(adj, self.k, dim=-1)
        mask_topk = torch.zeros_like(adj)
        mask_topk.scatter_(-1, topk_indices, 1.0)
        final_mask = mask_threshold * mask_topk
        # 生成加权邻接矩阵 (保留梯度)
        # 仅对保留的边进行 Softmax
        zero_vec = -9e15 * torch.ones_like(adj)
        adj_masked = torch.where(final_mask > 0, adj, zero_vec)
        adj_final = F.softmax(adj_masked, dim=-1)
        # 计算图平滑度损失 (Graph Smoothness Loss / Dirichlet Energy)
        # L_smooth = sum_{i,j} A_ij ||x_i - x_j||^2
        # 计算节点对距离矩阵
        x_norm_for_loss = F.normalize(x, p=2, dim=-1)
        dist_matrix = torch.cdist(x_norm_for_loss, x_norm_for_loss).pow(2) # (B, N, N)
        smoothness_loss = torch.mean(torch.sum(adj_final * dist_matrix, dim=(1, 2)))

        return adj_final, smoothness_loss
    

# 图卷积模块
class TGCN(torch.nn.Module):
    def __init__(self, in_features, bn_features, out_features):
        super().__init__()
        self.channels = 62  # EEG通道数
        self.in_features = in_features #65*32
        self.bn_features = bn_features #64
        self.out_features = out_features #32
        self.graph_learner = IterativeGraphLearner(
            input_dim=bn_features,
            hidden_dim=64,
            num_heads=4,
            topk=10
        )
        self.bnlin = Linear(in_features, bn_features) 
        self.gconv = DenseSAGEConv(in_features, out_features)
        self.residual = Linear(in_features, out_features) if in_features != out_features else nn.Identity()

    def forward(self, x):
        xa = torch.tanh(self.bnlin(x))
        adj = torch.matmul(xa, xa.transpose(2, 1))
        adj_mask, smooth_loss = self.graph_learner(xa)
        B, N, _ = adj.shape
        identity = torch.eye(N, device=adj.device).unsqueeze(0).repeat(B, 1, 1)
        adj = torch.softmax(adj, dim=2)
        adj = adj * adj_mask.to(adj.device) 
        adj_final = torch.softmax(adj + identity * 2.0, dim=2)  
        x = F.relu(self.gconv(x, adj_final)) + self.residual(x)
        return x, adj, smooth_loss


# 主网络架构
class MVEEGNet(torch.nn.Module):
    def __init__(self, dim_in, dim_h, d_out, sz_layer, 
                 num_views, channels, num_class):
        super().__init__()
        # 初始化参数
        self.stride = 2
        self.sz_layer = sz_layer              # 每个视图的GCN层数
        self.sc_views = num_views             # 视图数
        self.gcn_layer = nn.ModuleList(
            self.channal_block(
                self.stride, dim_in, dim_h, d_out))
        self.layer_norm = nn.LayerNorm(channels, eps=1e-6)
        self.drop4 = Dropout(0.2)
        self.fml = FML(num_views, np.array([[channels * d_out] * num_views]).transpose(), num_class)
        self.linend = Linear(channels * d_out * num_views, num_class)

    def channal_block(self, stride, d_in, d_h, d_out):
        layer = []
        for v in range(self.sc_views):
            t_layer = []
            for l in range(self.sz_layer):
                if l == 0:
                    in_feats = int(d_in)
                    bn_feats = int(d_h)
                    out_feats = int(d_h // stride)
                    t_layer.append(TGCN(in_feats, bn_feats, out_feats))
                elif l == self.sz_layer - 1:
                    div = int(stride * 2 * 2)
                    t_layer.append(TGCN(int(d_h // div), int(d_h // div), int(d_out)))
                else:
                    t_layer.append(TGCN(int(d_h // stride), int(d_h // (stride * 2)), int(d_h // (stride * 2 * 2))))
                # t_layer.append(nn.Dropout(self.drop_rate))   
            layer.append(nn.ModuleList(t_layer))
        return layer

    def forward(self, source_data):
        train_x = dict()
        for v in range(self.sc_views):
            node_type = f'view_{v}'
            x_v = source_data[node_type].x
            batch_v = source_data[node_type].batch
            train_x[v], mask_ = to_dense_batch(x_v, batch_v)
        # ----------------- 特征提取流程 -----------------
        all_hs = []
        all_adjs = []
        total_smooth_loss = 0.0
        for i, layer in enumerate(self.gcn_layer):
            # x_view = x.clone()
            x_view = train_x[i].clone()
            # import pdb; pdb.set_trace()
            for gcn_layer in layer:
                x_view, adj, layer_loss = gcn_layer(x_view)
                total_smooth_loss += layer_loss
            all_hs.append(self.drop4(x_view))
            all_adjs.append(adj)
        out = torch.concat(all_hs, dim=-1)
        flat_features = out.reshape(train_x[0].size(0), -1)
        alpha_a, pred, u_fused = self.fml(all_hs)
        class_output = self.linend(flat_features)
        # import pdb; pdb.set_trace()
        return alpha_a, u_fused, pred, total_smooth_loss, all_adjs