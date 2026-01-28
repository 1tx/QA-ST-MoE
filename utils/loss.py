import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableWeightedLoss(nn.Module):
    """
    一个带有可学习、带约束权重的通用损失函数。
    增加了强制最小间隔约束，防止权重塌缩。
    """

    def __init__ ( self, args, loss_type, has_qc=True, delta=1.0, min_weight_gap=2.0):
        """
        Args:
            min_weight_gap (float): 权重等级之间强制保持的最小差距。
                                    值越大，高质量数据的权重相对于低质量数据就越大。
                                    建议值：1.0 ~ 5.0 之间。
        """
        super().__init__()
        self.loss_type = loss_type.upper()
        self.delta = delta
        self.has_qc = has_qc
        self.arg = args
        self.min_weight_gap = min_weight_gap  # 新增：最小间隔

        if self.has_qc:
            num_learnable_weights = 4
        else:
            num_learnable_weights = 1

        # 初始化参数：保持较小，让初始的差异主要由 min_weight_gap 决定
        self.raw_weights = nn.Parameter(torch.randn(num_learnable_weights) * 0.01)

        if self.loss_type not in ['MAE', 'MSE', 'HUBER']:
            raise ValueError("Unsupported loss_type. Choose from 'MAE', 'MSE', 'HUBER'")

    def forward ( self, predictions, targets, quality_flags, enable: bool ):
        device = predictions.device

        # --- 1. 计算所有样本的基础误差 (保持不变) ---
        if self.loss_type == 'MAE':
            base_errors = F.l1_loss(predictions, targets, reduction='none')
        elif self.loss_type == 'MSE':
            base_errors = F.mse_loss(predictions, targets, reduction='none')
        elif self.loss_type == 'HUBER':
            base_errors = F.huber_loss(predictions, targets, delta=self.delta, reduction='none')
        # --- 2. 根据 'enable' 标志获取样本权重 ---
        if enable:
            if self.has_qc:
                # === 核心修改区域 ===
                gap3 = self.min_weight_gap + F.softplus(self.raw_weights[3])
                w3 = gap3
                # Level 2
                gap2 = self.min_weight_gap + F.softplus(self.raw_weights[2])
                w2 = w3 + gap2
                # Level 1
                gap1 = self.min_weight_gap + F.softplus(self.raw_weights[1])
                w1 = w2 + gap1
                # Level 0 (最好的数据)
                gap0 = self.min_weight_gap + F.softplus(self.raw_weights[0])
                w0 = w1 + gap0
                # 缺失数据权重固定为 0
                w4 = torch.tensor(0.0, device=device)
                weights = torch.stack([w0, w1, w2, w3, w4])
            else:
                # 无 QC 情况的处理
                gap0 = self.min_weight_gap + F.softplus(self.raw_weights[0])
                w0 = gap0
                w1 = torch.tensor(0.0, device=device)
                weights = torch.stack([w0, w1])
            # 查找权重 (保持不变)
            safe_flags = torch.clamp(quality_flags.squeeze().long(), max=len(weights) - 1)
            sample_weights = weights[safe_flags]

        else:
            qc_flags_squeezed = quality_flags.squeeze().long()
            sample_weights = (qc_flags_squeezed == 0).float()
        while len(sample_weights.shape) < len(base_errors.shape):
            sample_weights = sample_weights.unsqueeze(-1)
        # --- 计算最终 Loss (保持不变) ---
        weighted_errors = base_errors * sample_weights
        total_loss = torch.sum(weighted_errors)
        num_weighted_samples = torch.sum(sample_weights)
        # 防止除零
        mean_weighted_loss = total_loss / (num_weighted_samples + 1e-8)
        return mean_weighted_loss