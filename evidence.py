import torch.nn as nn
import torch
import torch.nn.functional as F


class RCML(nn.Module):
    def __init__(self, num_views, dims, num_classes):
        super(RCML, self).__init__()
        self.num_views = num_views
        self.num_classes = num_classes
        self.EvidenceCollectors = nn.ModuleList(
            [EvidenceCollector(dims[i], self.num_classes) 
            for i in range(self.num_views)])
        
    def forward(self, X):
        evidences = dict()
        for v in range(self.num_views):
            x_flat = torch.flatten(X[v], start_dim=1)
            evidences[v] = self.EvidenceCollectors[v](x_flat)
        # 1. 转换为狄利克雷参数 alpha = e + 1
        alphas = [e + 1 for e in evidences.values()]
        # 2. 狄利克雷算子融合 (基于均值的证据聚合)
        # 平滑不同视图的极端值
        alpha_fused = torch.zeros_like(alphas[0])
        for a in alphas:
            alpha_fused += a
        # 减去重复计算的先验(1)，保持 alpha = e_total + 1 的形式
        alpha_fused = alpha_fused - (self.num_views - 1)
        
        # 3. 计算融合后的不确定性 u = K / S
        S_fused = torch.sum(alpha_fused, dim=1, keepdim=True)
        u_fused = self.num_classes / S_fused
        
        # 4. 计算预测概率 (狄利克雷分布的均值)
        # Prob = alpha / S
        prob_fused = alpha_fused / S_fused
        return evidences, prob_fused, u_fused


class EvidenceCollector(nn.Module):
    def __init__(self, dims, num_classes):
        super(EvidenceCollector, self).__init__()
        self.num_layers = len(dims)
        self.net = nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.net.append(nn.Linear(dims[i], dims[i + 1]))
            self.net.append(nn.ReLU())
            self.net.append(nn.Dropout(0.1))
        self.net.append(nn.Linear(dims[self.num_layers - 1], num_classes))
        self.net.append(nn.Softplus())

    def forward(self, x):
        h = self.net[0](x)
        for i in range(1, len(self.net)):
            h = self.net[i](h)
        return h
    
class ERML(nn.Module):
    def __init__(self, num_views, dims, num_classes):
        """
        Args:
            num_views (int): 视图数量
            dims (list of list): 每个视图的网络结构尺寸。
                                 例如 [[in_dim1, hidden1], [in_dim2, hidden2]]
            num_classes (int): 分类类别数
        """
        super(ERML, self).__init__()
        self.num_views = num_views
        self.num_classes = num_classes
        
        # 1. 构建每个真实视图的 EvidenceCollector
        self.EvidenceCollectors = nn.ModuleList([
            EvidenceCollector(dims[i], self.num_classes) for i in range(self.num_views)
        ])
        
        # 2. 构建伪视图 (Pseudo-View) 的 EvidenceCollector
        # 伪视图的输入是所有真实视图输入的拼接 (假设所有输入都已经展平或在dim=1拼接)
        # dims[i] 代表第 i 个视图的输入层维度
        pseudo_dims = sum([dims[i] for i in range(self.num_views)])
        # 伪视图的隐藏层结构，这里默认复用第一个视图的隐藏层结构 dims[0][1:]
        # 你也可以根据需要自定义伪视图的隐藏层
        self.PseudoCollector = EvidenceCollector(pseudo_dims, self.num_classes)

    def DS_Combin_two(self, alpha1, alpha2):
        """
        Dempster's Rule of Combination for two evidences (Alpha parameters).
        """
        alpha = dict()
        alpha[0], alpha[1] = alpha1, alpha2
        b, S, E, u = dict(), dict(), dict(), dict()
        
        # 计算 Belief (b) 和 Uncertainty (u)
        for v in range(2):
            S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
            E[v] = alpha[v] - 1
            b[v] = E[v] / (S[v].expand(E[v].shape))
            u[v] = self.num_classes / S[v]

        # b^0 @ b^(0+1)
        bb = torch.bmm(b[0].view(-1, self.num_classes, 1), b[1].view(-1, 1, self.num_classes))
        
        # b^0 * u^1
        uv1_expand = u[1].expand(b[0].shape)
        bu = torch.mul(b[0], uv1_expand)
        
        # b^1 * u^0
        uv_expand = u[0].expand(b[0].shape)
        ub = torch.mul(b[1], uv_expand)
        
        # Calculate conflict K
        bb_sum = torch.sum(bb, dim=(1, 2), out=None)
        bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
        K = bb_sum - bb_diag

        # Calculate new b and u
        b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - K).view(-1, 1).expand(b[0].shape))
        u_a = torch.mul(u[0], u[1]) / ((1 - K).view(-1, 1).expand(u[0].shape))

        # Calculate new S and alpha
        S_a = self.num_classes / u_a
        e_a = torch.mul(b_a, S_a.expand(b_a.shape))
        alpha_a = e_a + 1
        return alpha_a

    def forward(self, X):
        """
        Args:
            X (list): 包含 num_views 个 Tensor 的列表，每个 Tensor 代表一个视图的数据。
        Returns:
            evidences (dict): 每个单独视图的 Evidence
            evidence_a (tensor): 所有真实视图融合后的 Evidence
            evidence_b (tensor): 真实视图融合结果 + 伪视图融合后的 Evidence
        """
        # 确保输入数据长度匹配
        assert len(X) == self.num_views
        # ------------ 1. 处理真实视图 ------------
        alphas = []
        for v in range(self.num_views):
            # 获取输入并展平（用于后续伪视图拼接）
            x_flat = torch.flatten(X[v], start_dim=1)
            # 计算当前视图的 Evidence
            ev_cls = self.EvidenceCollectors[v](x_flat)
            ev = F.softplus(ev_cls)
            alphas.append(ev + 1)
        # ------------ 2. 视图融合，两种方式1）输出证据后融合 2）先融合后输出证据 ------------
        # 1）输出证据后融合。递归两两融合
        alpha_a = alphas[0]
        for i in range(1, self.num_views):
            alpha_a = self.DS_Combin_two(alpha_a, alphas[i])
        # 2）先融合后输出证据
        # 拼接所有视图的 Flatten 特征
        # pseudo_view = torch.cat(X, dim=1).reshape(X[0].size(0), -1)
        # 计算伪视图 Evidence
        # fusion_ev = self.PseudoCollector(pseudo_view)
        # pse_alpha = F.softplus(fusion_ev)
        # fusion_alpha = pse_alpha + 1
        # ------------ 4. 最终融合 (得到 Evidence B) ------------
        # 将“真实视图的联合 Alpha”与“伪视图 Alpha”进行融合
        # alpha_b = self.DS_Combin_two(alpha_a, fusion_alpha)
        return alphas, alpha_a


class SANA(nn.Module):

    def __init__(self, h_dim, d_in, att_layers, dropout, temperature=1.0):
        super(SANA, self).__init__()
        """"""
        self.temperature = temperature ** 2
        self.lins = []
        for i in range(att_layers):
            if i == 0:
                self.lins.append(torch.nn.Linear(2 * d_in, h_dim))
            else:
                self.lins.append(torch.nn.Linear(h_dim, h_dim))
        self.trans_lin = torch.nn.ModuleList(self.lins)
        self.att_lin = torch.nn.Linear(h_dim + d_in, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.reset_parameters()

    def reset_parameters(self):
        for m in [self.att_lin]:
            m.reset_parameters()
        for m in self.trans_lin:
            m.reset_parameters()

    def forward(self, h_l, s_l):
        h1, h2 = h_l, s_l 
        z = torch.cat([h1, h2], dim=-1)
        for m in self.trans_lin:
            z = F.relu(self.dropout(m(z)))
        al1 = self.att_lin(torch.cat([z, h1], dim=-1))
        al2 = self.att_lin(torch.cat([z, h2], dim=-1))
        att = F.softmax(torch.cat([al1 / self.temperature + 1e-8, al2 / self.temperature + 1e-8], dim=-1), dim=-1)
        # print(att)
        alpha1 = att[..., 0:1]  # [B, N, 1]
        alpha2 = att[..., 1:2]  # [B, N, 1]
        out = alpha1 * h1 + alpha2 * h2
        out = self.layer_norm(out)
        return out


class SimilarMeasure():
    """"""

    def __init__(self, agg="sum"):
        super(SimilarMeasure, self).__init__()
        """"""
        self.agg = agg

    def correlation_loss(self, z: torch.tensor):
        """"""
        # sz_c,d = z.size(0),z.size(-1)
        z_ = z.mean(dim=-1, keepdim=True)
        z_stds = z.std(dim=-1)
        all_t = None
        us = None
        for m in (z - z_):
            if all_t is None:
                all_t = m
            else:
                all_t = all_t * m
        for u in z_stds:
            if us is None:
                us = u
            else:
                us = us * u
        sim = all_t.sum(dim=-1) / (z.size(-1) - 1) / us
        # sim_dis = 1 - torch.abs(sim)
        # return sim
        if self.agg == 'sum':
            return torch.sum(sim ** 2)
        else:
            return torch.mean(sim ** 2)
        

class FuzzyFusionLayer(nn.Module):
    def __init__(self, num_views, num_classes):
        super(FuzzyFusionLayer, self).__init__()
        self.num_views = num_views
        self.num_classes = num_classes
        
        # 模糊权重：学习每个视图在模糊推理中的重要性
        self.fuzzy_weights = nn.Parameter(torch.ones(num_views))
        # 隶属度微调层
        self.membership_scaling = nn.Parameter(torch.ones(num_classes))

    def forward(self, view_features_list):
        """
        view_features_list: 每个视图经过分类头后的 Logits 列表 [Batch, K]
        """
        # 1. 模糊化 (Fuzzification)
        # 将原始得分转换为隶属度 (Membership Degrees)
        memberships = [F.softmax(f, dim=1) for f in view_features_list]
        
        # 2. 模糊推理 (Fuzzy Inference - 使用加权代数乘积算子)
        # 归一化权重
        w = F.softmax(self.fuzzy_weights, dim=0)
        
        # 初始化融合后的隶属度 (用 1 开始，因为是乘积)
        fused_membership = torch.ones_like(memberships[0])
        
        for v in range(self.num_views):
            # 对每个视图的隶属度进行加权提升
            # 权重越高，该视图对乘积结果的影响越可控
            weighted_v = torch.pow(memberships[v], w[v])
            fused_membership = fused_membership * weighted_v
            
        # 3. 去模糊化 (Defuzzification)
        # 通过归一化重新获得类别的概率分布
        output_probs = F.normalize(fused_membership, p=1, dim=1)
        
        return output_probs