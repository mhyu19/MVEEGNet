import torch.nn as nn
import torch
import torch.nn.functional as F


class FML(nn.Module):
    def __init__(self, num_views, dims, num_classes):
        super(FML, self).__init__()
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
        # import pdb; pdb.set_trace()
        return evidences, prob_fused, u_fused


class EvidenceCollector(nn.Module):
    def __init__(self, dim, num_classes):
        super(EvidenceCollector, self).__init__()
        self.net = nn.ModuleList()
        self.net.append(nn.Linear(dim, num_classes))
        self.net.append(nn.Softplus())

    def forward(self, x):
        h = self.net[0](x)
        for i in range(1, len(self.net)):
            h = self.net[i](h)
        return h


class RCML(nn.Module):
    def __init__(self, num_views, dims, num_classes):
        super(RCML, self).__init__()
        self.num_views = num_views
        self.num_classes = num_classes
        # self.EvidenceCollectors = nn.ModuleList(
            # [EvidenceCollector(dim_in, dim_out, self.num_classes) for i in range(self.num_views)])
        self.EvidenceCollectors = nn.ModuleList([EvidenceCollector(dims[i], self.num_classes) for i in range(self.num_views)])

    def forward(self, X):
        # get evidence
        evidences = dict()
        for v in range(self.num_views):
            x_flat = torch.flatten(X[v], start_dim=1)
            evidences[v] = self.EvidenceCollectors[v](x_flat)
        # average belief fusion
        evidence_a = evidences[0]
        for i in range(1, self.num_views):
            evidence_a = (evidences[i] + evidence_a) / 2
        return evidences, evidence_a


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
        evidences = dict()
        alphas = []
        flat_inputs = []

        for v in range(self.num_views):
            # 获取输入并展平（用于后续伪视图拼接）
            x_flat = torch.flatten(X[v], start_dim=1)
            flat_inputs.append(x_flat)
            
            # 计算当前视图的 Evidence
            ev = self.EvidenceCollectors[v](x_flat)
            evidences[v] = ev
            
            # 转换为 Alpha (Alpha = Evidence + 1) 用于 DS 融合
            alphas.append(ev + 1)

        # ------------ 2. 融合真实视图 (得到 Evidence A) ------------
        # 递归两两融合
        alpha_a = alphas[0]
        for i in range(1, self.num_views):
            alpha_a = self.DS_Combin_two(alpha_a, alphas[i])
        
        # 将融合后的 Alpha 转回 Evidence (Evidence = Alpha - 1)
        evidence_a = alpha_a - 1
        return evidences, evidence_a
        
