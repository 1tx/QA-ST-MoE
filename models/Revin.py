import torch
import torch.nn as nn


class RevIN(nn.Module):
    def __init__ ( self, num_features: int, eps=1e-5, affine=True ):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def _init_params ( self ):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward ( self, x, mode: str, stats=None ):
        if mode == 'norm':
            if stats is None:
                stats = self._get_statistics(x)
            x = self._normalize(x, stats)
            return x, stats
        elif mode == 'denorm':
            if stats is None:
                raise ValueError("Statistics must be provided for de-normalization.")
            x = self._denormalize(x, stats)
            return x
        else:
            raise NotImplementedError

    def _get_statistics ( self, x ):
        dim2reduce = tuple(range(1, x.ndim - 1))
        mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
        return mean, stdev

    def _normalize ( self, x, stats ):
        mean, stdev = stats
        x = x - mean
        x = x / stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize ( self, x, stats ):
        mean, stdev = stats
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * stdev
        x = x + mean
        return x


class HybridRevIN(nn.Module):
    def __init__ ( self, num_features, global_mean=None, global_std=None, threshold=0.5, eps=1e-5, affine=True ):
        """
        Args:
            num_features: 特征维度
            global_mean: 全局均值 (float 或 list)，若为None则默认为0
            global_std: 全局标准差 (float 或 list)，若为None则默认为1
            threshold: 有效率阈值 (0~1)，超过此比例才计算实例统计量
            eps: 防止除0
            affine: 是否使用可学习仿射变换
        """
        super().__init__()
        self.num_features = num_features
        self.threshold = threshold
        self.eps = eps
        self.affine = affine
        if global_mean is None: global_mean = torch.zeros(num_features)
        if global_std is None: global_std = torch.ones(num_features)
        g_mean = torch.as_tensor(global_mean, dtype=torch.float32)
        if g_mean.ndim == 0:
            g_mean = g_mean.unsqueeze(0)  # 强制转为 [1]
        g_std = torch.as_tensor(global_std, dtype=torch.float32)
        if g_std.ndim == 0:
            g_std = g_std.unsqueeze(0)  # 强制转为 [1]
        self.register_buffer('global_mean', g_mean)
        self.register_buffer('global_std', g_std)
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def _get_statistics ( self, x, stats_mask):
        # 1. 计算每个样本的有效率 (B, 1, 1)
        # mask 是 [B, L, 1]，该部分会将为1的数量进行求和，得到有效数据的数量
        valid_count = stats_mask.sum(dim=1, keepdim=True)  # [B, 1, 1]
        total_len = x.shape[1]
        valid_ratio = valid_count / total_len
        # 2. 是否使用实例统计量的判断掩码 [B, 1, 1]
        # 如果 ratio > threshold，则为 True
        use_instance_stats = (valid_ratio > self.threshold)
        # -------------------------------------------------------
        # A. 计算实例统计量 (Instance Stats)
        # 即使 count 为 0 也先算出一个数，反正后面会被 where 过滤掉
        safe_count = valid_count.clamp(min=1.0)
        sum_val = (x * stats_mask).sum(dim=1, keepdim=True)
        inst_mean = sum_val / safe_count
        var_sum = ((x - inst_mean) ** 2 * stats_mask).sum(dim=1, keepdim=True)
        inst_var = var_sum / safe_count
        inst_std = torch.sqrt(inst_var + self.eps)
        glob_mean = self.global_mean.view(1, 1, -1).expand_as(inst_mean)
        glob_std = self.global_std.view(1, 1, -1).expand_as(inst_std)
        # -------------------------------------------------------
        # C. 融合 (Hybrid Selection)
        # -------------------------------------------------------
        # torch.where(condition, x, y) -> condition为True选x，否则选y
        final_mean = torch.where(use_instance_stats, inst_mean, glob_mean)
        final_std = torch.where(use_instance_stats, inst_std, glob_std)
        return final_mean, final_std

    def forward ( self, x, mode, stats_mask=None,keep_mask=None, stats=None ):
        if mode == 'norm':
            if stats_mask is None: stats_mask = torch.ones_like(x)
            if keep_mask is None: keep_mask = stats_mask
            # 获取混合统计量
            mean, stdev = self._get_statistics(x, stats_mask)
            # 归一化
            x_norm = (x - mean) / stdev
            # 再次把无效区域置0 (防止脏数据进入网络)
            x_norm = x_norm * keep_mask
            if self.affine:
                x_norm = x_norm * self.affine_weight + self.affine_bias
            return x_norm, (mean, stdev)
        elif mode == 'denorm':
            if stats is None: return x
            mean, stdev = stats
            x_denorm = x
            if self.affine:
                x_denorm = (x_denorm - self.affine_bias) / (self.affine_weight + self.eps)
            x_denorm = x_denorm * stdev + mean
            return x_denorm
