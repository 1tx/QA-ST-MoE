from .MOE import MoETransformer
from .Embeding import CombinedEmbedding, TimeEmbedding, SiteEmbedding, NIGHT_Embedding
from .Revin import RevIN, HybridRevIN
import torch
import torch.nn as nn
import torch.nn.functional as F
from .MOE import Transformer_base


class AttentionPooling(nn.Module):
    """
    通过一个小型MLP计算每个时间步的权重，然后进行加权求和。
    """

    def __init__ ( self, hidden_dim ):
        super().__init__()
        # 定义用于计算注意力分数的小型神经网络
        # Linear(hidden_dim -> hidden_dim/2) -> Tanh -> Linear(hidden_dim/2 -> 1)
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward ( self, x ):
        # x shape: [B, seq_len, hidden_dim]
        # 1. 计算每个时间步的注意力分数
        # att_scores shape: [B, seq_len, 1]
        att_scores = self.attention_net(x)
        # 2. 使用 softmax 将分数归一化为权重
        # weights shape: [B, seq_len, 1]
        # softmax 必须在 seq_len 维度上进行，确保每个序列的权重和为1
        att_weights = F.softmax(att_scores, dim=1)
        # 3. 进行加权求和
        # ( [B, seq_len, 1] * [B, seq_len, hidden_dim] ) -> [B, seq_len, hidden_dim]
        # 然后在 seq_len 维度上求和 -> [B, hidden_dim]
        context_vector = torch.sum(att_weights * x, dim=1)
        return context_vector


class GapfillModel(nn.Module):
    def __init__ ( self, args, unique_sites_num, unique_igbp_num, mode ):
        super().__init__()
        self.args = args
        self.hidden_dim = self.args.hidden_dim
        self.embed_dim = self.args.embed_dim
        self.revin1 = RevIN(len(args.aux_variables))
        g_mean = getattr(args, 'default_mean', [0.0])
        g_std = getattr(args, 'default_std', [1.0])
        self.revin2 = HybridRevIN(
            num_features=1,
            global_mean=g_mean,
            global_std=g_std,
            threshold=0.5,
            affine=True
        )
        self.embedder = CombinedEmbedding(args, unique_sites_num, unique_igbp_num)
        self.moe_transformer = MoETransformer(args)
        self.mode = mode
        self.attention_pooling = AttentionPooling(self.hidden_dim)
        self.current_time_embed = TimeEmbedding(self.embed_dim)
        self.current_site_embed = SiteEmbedding(unique_sites_num, unique_igbp_num, self.embed_dim)
        self.current_driver_proj = nn.Linear(len(args.aux_variables), self.embed_dim)
        current_info_dim = self.embed_dim * 3  # (时间 + 站点 + 连续驱动)
        self.current_info_projection = nn.Sequential(
            nn.Linear(current_info_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim)
        )
        self.current_night_embed = NIGHT_Embedding(self.hidden_dim)
        total_context_dim = self.hidden_dim * 2
        self.output_projection = nn.Linear(total_context_dim, self.args.output_dim)

    def forward ( self, input_data ):
        driver_history = input_data['driver_history']  # shape[B,seq_len,feature_num]
        driver_current = input_data['driver_current']
        history_value = input_data['input_values']  # Shape: [B, seq_len, 1]
        input_quality = input_data["input_quality"]  # Shape: [B, seq_len, 1]
        night_mask = input_data["night_mask"]  # shape [B,1]
        part_to_normalize = driver_history[..., :self.args.feature_in]
        part_to_keep = driver_history[..., self.args.feature_in:]
        normalized_part, driver_stats = self.revin1(part_to_normalize, 'norm')
        driver = torch.cat([normalized_part, part_to_keep], dim=-1)
        # 该部分会将input_quality部分值为0的部分提取出来，值为0的valid_mask 值会为1(True)
        stats_mask = (input_quality == 0).float()
        if stats_mask.dim() == 2: stats_mask = stats_mask.unsqueeze(-1)
        keep_mask = (input_quality != 4).float()
        if keep_mask.dim() == 2: keep_mask = keep_mask.unsqueeze(-1)
        # 传入 mask，HybridRevIN 会自动判断使用 Instance 还是 Global Stats
        history_value_norm, stats = self.revin2(history_value, 'norm', stats_mask=stats_mask, keep_mask=keep_mask)
        x_embed_history = self.embedder(driver, history_value, input_quality)
        # 对当前状态下的driver进行嵌入
        driver_cont_current = driver_current[..., :len(self.args.aux_variables)]
        meta_data_current = driver_current[..., len(self.args.aux_variables):].long()  # [B, 4]
        tod_c, doy_c, site_id_c, igbp_c = [meta_data_current[..., i] for i in range(4)]  # [B]
        # 复用 RevIN 参数对当前时刻 Driver 进行归一化
        driver_cont_current_unsqueezed = driver_cont_current.unsqueeze(1)
        normalized_driver_current_unsqueezed, _ = self.revin1(driver_cont_current_unsqueezed, 'norm',
                                                              stats=driver_stats)
        normalized_driver_current = normalized_driver_current_unsqueezed.squeeze(1)
        # 嵌入当前特征
        time_e_c = self.current_time_embed(tod_c, doy_c)  # [B, E]
        site_e_c = self.current_site_embed(site_id_c, igbp_c)  # [B, E]
        driver_e_c = self.current_driver_proj( normalized_driver_current)  # [B, E]
        # 拼接并投影
        combined_current_features = torch.cat([time_e_c, site_e_c, driver_e_c], dim=-1)
        H_current = self.current_info_projection(combined_current_features)
        H_night_bias = self.current_night_embed(night_mask).squeeze(1)
        H_current_with_night = H_current + H_night_bias
        # 融合 H_past 和 H_current
        H_current_token = H_current_with_night.unsqueeze(1)
        x_input = torch.cat([x_embed_history, H_current_token], dim=1)
        hidden_states = self.moe_transformer(x_input)
        # 1. 拆分 Output
        seq_len = x_embed_history.shape[1]
        hist_out_states = hidden_states[:, :seq_len, :]
        curr_out_state = hidden_states[:, -1, :]
        # 取出属于当前时刻的输出 -> [B, hidden_dim]
        curr_out_state = hidden_states[:, -1, :]
        H_past_pooled = self.attention_pooling(hist_out_states)
        combined_context = torch.cat([H_past_pooled, curr_out_state], dim=-1)
        # 送入修改后的输出层
        out = self.output_projection(combined_context)
        out_3d = out.unsqueeze(1)
        out_physical_3d = self.revin2(out_3d, "denorm", stats)
        out_physical = out_physical_3d.squeeze(1)
        out_corrected = self.apply_physical_correction(out_physical, night_mask)
        return out_corrected

    def apply_physical_correction ( self, out_physical, night_mask ):
        """
        在 *物理空间* 中应用您在 output_projection 中定义的特定物理约束。

        Args:
            out_physical (torch.Tensor):
            night_mask (torch.Tensor): 夜间标记 [B, 1] (假设 1=夜晚)
        Returns:
            torch.Tensor: 经过校正的输出
        """
        clean_var = self.args.filling_var.strip()
        # 1. 应用主要的物理约束
        if clean_var in ['P', 'PA', 'SW_IN_F_MDS', 'LW_IN_F_MDS', 'VPD_F_MDS', 'WS']:
            output = F.softplus(out_physical)
        else:
            output = out_physical
        return output


class GapfillModel_NoRevin(nn.Module):
    def __init__ ( self, args, unique_sites_num, unique_igbp_num, mode ):
        super().__init__()
        self.args = args
        self.hidden_dim = self.args.hidden_dim
        self.embed_dim = self.args.embed_dim
        self.embedder = CombinedEmbedding(args, unique_sites_num, unique_igbp_num)
        self.moe_transformer = MoETransformer(args)
        self.mode = mode
        self.attention_pooling = AttentionPooling(self.hidden_dim)
        self.current_time_embed = TimeEmbedding(self.embed_dim)
        self.current_site_embed = SiteEmbedding(unique_sites_num, unique_igbp_num, self.embed_dim)
        self.current_driver_proj = nn.Linear(len(args.aux_variables), self.embed_dim)
        current_info_dim = self.embed_dim * 3  # (时间 + 站点 + 连续驱动)
        self.current_info_projection = nn.Sequential(
            nn.Linear(current_info_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim)
        )
        self.current_night_embed = NIGHT_Embedding(self.hidden_dim)
        total_context_dim = self.hidden_dim * 2
        self.output_projection = nn.Linear(total_context_dim, self.args.output_dim)

    def forward ( self, input_data ):
        driver_history = input_data['driver_history']  # shape[B,seq_len,feature_num]
        driver_current = input_data['driver_current']
        history_value = input_data['input_values']  # Shape: [B, seq_len, 1]
        input_quality = input_data["input_quality"]  # Shape: [B, seq_len, 1]
        night_mask = input_data["night_mask"]  # shape [B,1]
        # 该部分会将input_quality部分值为0的部分提取出来，值为0的valid_mask 值会为1
        x_embed_history = self.embedder(driver_history, history_value, input_quality)
        hidden_state = self.moe_transformer(x_embed_history)
        H_past = self.attention_pooling(hidden_state)
        # 对当前状态下的driver进行嵌入
        driver_cont_current = driver_current[..., :len(self.args.aux_variables)]  # [B, D_cont]
        meta_data_current = driver_current[..., len(self.args.aux_variables):].long()  # [B, 4]
        tod_c, doy_c, site_id_c, igbp_c = [meta_data_current[..., i] for i in range(4)]  # [B]
        # 嵌入当前特征
        time_e_c = self.current_time_embed(tod_c, doy_c)  # [B, E]
        site_e_c = self.current_site_embed(site_id_c, igbp_c)  # [B, E]
        driver_e_c = self.current_driver_proj(driver_cont_current)  # [B, E]
        # 拼接并投影
        combined_current_features = torch.cat([time_e_c, site_e_c, driver_e_c], dim=-1)
        H_current = self.current_info_projection(combined_current_features)
        H_night_bias = self.current_night_embed(night_mask).squeeze(1)
        H_current_with_night = H_current + H_night_bias
        # 融合 H_past 和 H_current
        combined_context = torch.cat([H_past, H_current_with_night], dim=-1)  # Shape: [B, hidden_dim * 2]
        # 送入修改后的输出层
        out = self.output_projection(combined_context)
        out_3d = out.unsqueeze(1)
        out_physical = out_3d.squeeze(1)
        out_corrected = self.apply_physical_correction(out_physical, night_mask)
        return out_corrected

    def apply_physical_correction ( self, out_physical, night_mask ):
        """
        在 *物理空间* 中应用您在 output_projection 中定义的特定物理约束。

        Args:
            out_physical (torch.Tensor):
            night_mask (torch.Tensor): 夜间标记 [B, 1] (假设 1=夜晚)
        Returns:
            torch.Tensor: 经过校正的输出
        """
        clean_var = self.args.filling_var.strip()
        # 1. 应用主要的物理约束
        if clean_var in ['P', 'PA', 'SW_IN_F_MDS', 'LW_IN_F_MDS', 'VPD_F_MDS', 'WS']:
            output = F.softplus(out_physical)
        else:
            output = out_physical
        return output


class Transformer_NoEmbed(nn.Module):
    def __init__ ( self, args, mode ):
        super().__init__()
        self.args = args
        self.hidden_dim = self.args.hidden_dim
        self.mode = mode
        # 1. Driver 的 RevIN (针对连续气象变量)
        self.revin1 = RevIN(len(args.aux_variables))
        # 2. History Value 的 HybridRevIN (核心修改)
        g_mean = getattr(args, 'default_mean', [0.0])
        g_std = getattr(args, 'default_std', [1.0])
        self.revin2 = HybridRevIN(
            num_features=1,
            global_mean=g_mean,
            global_std=g_std,
            threshold=0.5,  # 设定有效率阈值，例如 50%
            affine=True
        )
        # 计算输入维度
        # driver (feature_num) + history_value (1)
        self.input_continuous_dim = len(args.aux_variables) + 1
        # 替代 CombinedEmbedding 的线性投影
        self.history_projection = nn.Linear(self.input_continuous_dim, self.hidden_dim)
        # Transformer 骨干
        self.moe_transformer = Transformer_base(args)
        self.attention_pooling = AttentionPooling(self.hidden_dim)
        # Current 分支投影
        self.current_driver_proj = nn.Linear(len(args.aux_variables), self.hidden_dim)
        self.current_info_projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim)
        )
        # 输出层
        total_context_dim = self.hidden_dim * 2
        self.output_projection = nn.Linear(total_context_dim, self.args.output_dim)

    def forward ( self, input_data ):
        # --- 1. 数据提取 ---
        driver_history = input_data['driver_history']  # [B, seq_len, feature_num]
        driver_current = input_data['driver_current']
        history_value = input_data['input_values']  # [B, seq_len, 1]
        input_quality = input_data['input_quality']  # [B, seq_len, 1] (假设 1=Good/Filled, 0=Missing)
        night_mask = input_data['night_mask']
        # --- 2. 归一化处理 ---
        # A. Driver 归一化 (修正逻辑：需保留未归一化的部分)
        part_to_normalize = driver_history[..., :self.args.feature_in]
        driver_history, _ = self.revin1(part_to_normalize, 'norm')  # revin1 返回 (x, stats)
        # B. History Value 归一化 (使用 HybridRevIN)
        # 制作掩码: 确保是 float 类型，维度 [B, L, 1]
        valid_mask = (input_quality == 0).float()
        if valid_mask.dim() == 2:
            valid_mask = valid_mask.unsqueeze(-1)
        # 传入 mask，HybridRevIN 会自动判断使用 Instance 还是 Global Stats
        history_value_norm, stats = self.revin2(history_value, 'norm', mask=valid_mask)
        # --- 3. 历史分支投影 ---
        # 拼接连续特征: [Normalized_Driver, Normalized_Value]
        # 维度: [B, seq_len, driver_dim + 1]
        combined_input = torch.cat([driver_history, history_value_norm], dim=-1)
        # 线性投影
        x_embed_history = self.history_projection(combined_input)
        # --- 4. Transformer 编码 ---
        hidden_state = self.moe_transformer(x_embed_history)
        H_past = self.attention_pooling(hidden_state)
        # --- 5. 当前分支投影 ---
        # 仅提取连续驱动变量
        driver_cont_current = driver_current[..., :len(self.args.aux_variables)]
        # 线性投影
        H_current_driver = self.current_driver_proj(driver_cont_current)
        H_current = self.current_info_projection(H_current_driver)

        # --- 6. 融合与输出 ---
        combined_context = torch.cat([H_past, H_current], dim=-1)
        out = self.output_projection(combined_context)

        # --- 7. 反归一化 ---
        out_3d = out.unsqueeze(1)
        # 使用 HybridRevIN 进行反归一化 (传入之前计算出的 stats)
        out_physical_3d = self.revin2(out_3d, "denorm", stats=stats)
        out_physical = out_physical_3d.squeeze(1)
        # --- 8. 物理校正 ---
        out_corrected = self.apply_physical_correction(out_physical, night_mask)
        return out_corrected

    def apply_physical_correction ( self, out_physical, night_mask ):
        clean_var = self.args.filling_var.strip()
        if clean_var in ['P', 'PA', 'SW_IN_F_MDS', 'LW_IN_F_MDS', 'VPD_F_MDS', 'WS']:
            output = F.softplus(out_physical)
        else:
            output = out_physical
        return output


class Transformer_GapfillModel(nn.Module):
    def __init__ ( self, args, unique_sites_num, unique_igbp_num, mode ):
        super().__init__()
        self.args = args
        self.hidden_dim = self.args.hidden_dim
        self.embed_dim = self.args.embed_dim
        self.revin1 = RevIN(len(args.aux_variables))
        g_mean = getattr(args, 'default_mean', [0.0])
        g_std = getattr(args, 'default_std', [1.0])
        self.revin2 = HybridRevIN(
            num_features=1,
            global_mean=g_mean,
            global_std=g_std,
            threshold=0.5,
            affine=True
        )
        self.embedder = CombinedEmbedding(args, unique_sites_num, unique_igbp_num)
        self.transformer = Transformer_base(args)
        self.mode = mode
        self.attention_pooling = AttentionPooling(self.hidden_dim)
        self.current_time_embed = TimeEmbedding(self.embed_dim)
        self.current_site_embed = SiteEmbedding(unique_sites_num, unique_igbp_num, self.embed_dim)
        self.current_driver_proj = nn.Linear(len(args.aux_variables), self.embed_dim)
        current_info_dim = self.embed_dim * 3  # (时间 + 站点 + 连续驱动)
        self.current_info_projection = nn.Sequential(
            nn.Linear(current_info_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim)
        )
        self.current_night_embed = NIGHT_Embedding(self.hidden_dim)
        total_context_dim = self.hidden_dim * 2
        self.output_projection = nn.Linear(total_context_dim, self.args.output_dim)

    def forward ( self, input_data ):
        driver_history = input_data['driver_history']  # shape[B,seq_len,feature_num]
        driver_current = input_data['driver_current']
        history_value = input_data['input_values']  # Shape: [B, seq_len, 1]
        input_quality = input_data["input_quality"]  # Shape: [B, seq_len, 1]
        night_mask = input_data["night_mask"]  # shape [B,1]
        part_to_normalize = driver_history[..., :self.args.feature_in]
        part_to_keep = driver_history[..., self.args.feature_in:]
        normalized_part, _ = self.revin1(part_to_normalize, 'norm')
        driver = torch.cat([normalized_part, part_to_keep], dim=-1)
        # 该部分会将input_quality部分值为0的部分提取出来，值为0的valid_mask 值会为1
        valid_mask = (input_quality == 0).float()
        if valid_mask.dim() == 2:
            valid_mask = valid_mask.unsqueeze(-1)
        # 传入 mask，HybridRevIN 会自动判断使用 Instance 还是 Global Stats
        history_value_norm, stats = self.revin2(history_value, 'norm', mask=valid_mask)
        x_embed_history = self.embedder(driver, history_value, input_quality)
        hidden_state = self.transformer(x_embed_history)
        H_past = self.attention_pooling(hidden_state)
        # 对当前状态下的driver进行嵌入
        driver_cont_current = driver_current[..., :len(self.args.aux_variables)]  # [B, D_cont]
        meta_data_current = driver_current[..., len(self.args.aux_variables):].long()  # [B, 4]
        tod_c, doy_c, site_id_c, igbp_c = [meta_data_current[..., i] for i in range(4)]  # [B]
        # 嵌入当前特征
        time_e_c = self.current_time_embed(tod_c, doy_c)  # [B, E]
        site_e_c = self.current_site_embed(site_id_c, igbp_c)  # [B, E]
        driver_e_c = self.current_driver_proj(driver_cont_current)  # [B, E]
        # 拼接并投影
        combined_current_features = torch.cat([time_e_c, site_e_c, driver_e_c], dim=-1)
        H_current = self.current_info_projection(combined_current_features)
        H_night_bias = self.current_night_embed(night_mask).squeeze(1)
        H_current_with_night = H_current + H_night_bias
        # 融合 H_past 和 H_current
        combined_context = torch.cat([H_past, H_current_with_night], dim=-1)  # Shape: [B, hidden_dim * 2]
        # 送入修改后的输出层
        out = self.output_projection(combined_context)
        out_3d = out.unsqueeze(1)
        out_physical_3d = self.revin2(out_3d, "denorm", stats)
        out_physical = out_physical_3d.squeeze(1)
        out_corrected = self.apply_physical_correction(out_physical, night_mask)
        return out_corrected

    def apply_physical_correction ( self, out_physical, night_mask ):
        """
        在 *物理空间* 中应用您在 output_projection 中定义的特定物理约束。

        Args:
            out_physical (torch.Tensor):
            night_mask (torch.Tensor): 夜间标记 [B, 1] (假设 1=夜晚)
        Returns:
            torch.Tensor: 经过校正的输出
        """
        clean_var = self.args.filling_var.strip()
        # 1. 应用主要的物理约束
        if clean_var in ['P', 'PA', 'SW_IN_F_MDS', 'LW_IN_F_MDS', 'VPD_F_MDS', 'WS']:
            output = F.softplus(out_physical)
        else:
            output = out_physical
        return output


class GapfillModel_NoEmbed(nn.Module):
    def __init__ ( self, args, mode ):
        super().__init__()
        self.args = args
        self.hidden_dim = self.args.hidden_dim
        self.mode = mode
        # 1. Driver 的 RevIN (针对连续气象变量)
        self.revin1 = RevIN(len(args.aux_variables))
        # 2. History Value 的 HybridRevIN (核心修改)
        g_mean = getattr(args, 'default_mean', [0.0])
        g_std = getattr(args, 'default_std', [1.0])
        self.revin2 = HybridRevIN(
            num_features=1,
            global_mean=g_mean,
            global_std=g_std,
            threshold=0.5,  # 设定有效率阈值，例如 50%
            affine=True
        )
        # 计算输入维度
        # driver (feature_num) + history_value (1)
        self.input_continuous_dim = len(args.aux_variables) + 1
        # 替代 CombinedEmbedding 的线性投影
        self.history_projection = nn.Linear(self.input_continuous_dim, self.hidden_dim)
        # Transformer 骨干
        self.moe_transformer = MoETransformer(args)
        self.attention_pooling = AttentionPooling(self.hidden_dim)
        # Current 分支投影
        self.current_driver_proj = nn.Linear(len(args.aux_variables), self.hidden_dim)
        self.current_info_projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim)
        )
        # 输出层
        total_context_dim = self.hidden_dim * 2
        self.output_projection = nn.Linear(total_context_dim, self.args.output_dim)

    def forward ( self, input_data ):
        # --- 1. 数据提取 ---
        driver_history = input_data['driver_history']  # [B, seq_len, feature_num]
        driver_current = input_data['driver_current']
        history_value = input_data['input_values']  # [B, seq_len, 1]
        input_quality = input_data['input_quality']  # [B, seq_len, 1] (假设 1=Good/Filled, 0=Missing)
        night_mask = input_data['night_mask']
        # --- 2. 归一化处理 ---
        # A. Driver 归一化 (修正逻辑：需保留未归一化的部分)
        part_to_normalize = driver_history[..., :self.args.feature_in]
        driver_history, _ = self.revin1(part_to_normalize, 'norm')  # revin1 返回 (x, stats)
        # B. History Value 归一化 (使用 HybridRevIN)
        # 制作掩码: 确保是 float 类型，维度 [B, L, 1]
        valid_mask = (input_quality == 0).float()
        if valid_mask.dim() == 2:
            valid_mask = valid_mask.unsqueeze(-1)
        # 传入 mask，HybridRevIN 会自动判断使用 Instance 还是 Global Stats
        history_value_norm, stats = self.revin2(history_value, 'norm', mask=valid_mask)
        # --- 3. 历史分支投影 ---
        # 拼接连续特征: [Normalized_Driver, Normalized_Value]
        # 维度: [B, seq_len, driver_dim + 1]
        combined_input = torch.cat([driver_history, history_value_norm], dim=-1)
        # 线性投影
        x_embed_history = self.history_projection(combined_input)
        # --- 4. Transformer 编码 ---
        hidden_state = self.moe_transformer(x_embed_history)
        H_past = self.attention_pooling(hidden_state)
        # --- 5. 当前分支投影 ---
        # 仅提取连续驱动变量
        driver_cont_current = driver_current[..., :len(self.args.aux_variables)]
        # 线性投影
        H_current_driver = self.current_driver_proj(driver_cont_current)
        H_current = self.current_info_projection(H_current_driver)

        # --- 6. 融合与输出 ---
        combined_context = torch.cat([H_past, H_current], dim=-1)
        out = self.output_projection(combined_context)

        # --- 7. 反归一化 ---
        out_3d = out.unsqueeze(1)
        # 使用 HybridRevIN 进行反归一化 (传入之前计算出的 stats)
        out_physical_3d = self.revin2(out_3d, "denorm", stats=stats)
        out_physical = out_physical_3d.squeeze(1)
        # --- 8. 物理校正 ---
        out_corrected = self.apply_physical_correction(out_physical, night_mask)
        return out_corrected

    def apply_physical_correction ( self, out_physical, night_mask ):
        clean_var = self.args.filling_var.strip()
        if clean_var in ['P', 'PA', 'SW_IN_F_MDS', 'LW_IN_F_MDS', 'VPD_F_MDS', 'WS']:
            output = F.softplus(out_physical)
        else:
            output = out_physical
        return output
