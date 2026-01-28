import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    """
    处理所有时间相关特征的嵌入。
    Handles embeddings for all time-related features.
    """

    def __init__ ( self, embed_dim ):
        super().__init__()
        # 建立 tod, doy, dow 的嵌入层
        self.tod_embed = nn.Embedding(48, embed_dim)  # Time of Day (0-47)
        self.doy_embed = nn.Embedding(366, embed_dim)  # Day of Year (0-365)

    def forward ( self, tod, doy):
        # 分别获取嵌入向量并将它们相加，融合成一个统一的时间表示
        tod_e = self.tod_embed(tod)
        doy_e = self.doy_embed(doy)
        return tod_e + doy_e


class NIGHT_Embedding(nn.Module):
    def __init__ ( self, embed_dim ):
        super().__init__()
        self.night_embed = nn.Embedding(3, embed_dim)

    def forward ( self, night_mask):

        return self.night_embed(night_mask)


class SiteEmbedding(nn.Module):
    """
    处理所有站点相关特征的嵌入。
    Handles embeddings for all site-related features.
    """

    def __init__ ( self, site_num, igbp_num, embed_dim ):
        super().__init__()
        self.site_embed = nn.Embedding(site_num, embed_dim)
        self.igbp_embed = nn.Embedding(igbp_num, embed_dim)

    def forward ( self, site_id, igbp):
        site_e = self.site_embed(site_id)
        igbp_e = self.igbp_embed(igbp)
        return site_e + igbp_e


class QualityEmbedding(nn.Module):
    """
    [NEW] Handles embeddings for data quality flags.
    """

    def __init__ ( self, args, embed_dim ):
        super().__init__()
        # 根据有无QC变量，设置不同的Quality_Embedding
        self.quality_embed = nn.Embedding(args.default_miss_flag + 1, embed_dim)

    def forward ( self, quality_flags ):
        return self.quality_embed(quality_flags)


# --- 最终的组合模块 ---

class CombinedEmbedding(nn.Module):
    """
    组合所有输入特征，为Transformer生成最终的输入张量。
    Combines all input features to generate the final input tensor for the Transformer.
    """

    def __init__ ( self, args, site_num, igbp_num ):
        super().__init__()
        embed_dim = args.embed_dim
        self.args = args
        # 1. 类别特征的嵌入层
        # Embeddings for categorical features
        self.site_embed = SiteEmbedding(site_num, igbp_num, embed_dim)
        self.time_embed = TimeEmbedding(embed_dim)
        self.quality_embed = QualityEmbedding(args, embed_dim)
        self.night_embed = NIGHT_Embedding(args.hidden_dim)
        # 2. 连续值特征的投影层
        # Linear projection layers for continuous features
        # 将 driver (ERA5-Land) 投影到 embed_dim
        self.driver_projection = nn.Linear(len(args.aux_variables), embed_dim)
        # 将 input_values (观测值自身) 投影到 embed_dim
        self.value_projection = nn.Linear(1, embed_dim)
        # 3. 最终组合投影层
        # 我们的总输入特征数 = 站点(1) + 时间(1) + 质量(1) + 驱动(1) + 观测值的历史序列 = 5
        total_embed_dim = embed_dim * 5
        self.final_projection = nn.Sequential(
            nn.Linear(total_embed_dim, args.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(args.hidden_dim)  # 使用 LayerNorm 增加稳定性
        )

    def forward ( self, driver,input_value,input_quality):
        """
        接收来自 Dataset 的字典输入，并进行处理。
        Accepts a dictionary of inputs from the Dataset.
        """
        # --- 2. 从 driver 张量中分离出不同的特征 ---
        # a. 分离出连续的驱动变量 (ERA5-Land等)
        driver_continuous = driver[..., :len(self.args.aux_variables)]
        # b. 分离出类别元数据
        meta_data = driver[..., len(self.args.aux_variables):].long()
        tod, doy, site_id, igbp = [meta_data[..., i] for i in range(4)]
        # --- 3. 分别进行嵌入和投影 ---
        # a. 类别特征嵌入
        site_e = self.site_embed(site_id, igbp)  # Shape: [B, seq_len,embed_dim]
        time_e = self.time_embed(tod, doy)  # Shape: [B, seq_len,embed_dim]
        quality_e = self.quality_embed(input_quality.squeeze(-1))  # Shape: [B, seq_len, embed_dim]
        # b. 连续特征投影
        driver_e = self.driver_projection(driver_continuous)  # Shape: [B, seq_len, embed_dim]
        value_e = self.value_projection(input_value)  # Shape: [B, seq_len, embed_dim]
        # --- 4. 拼接所有特征 ---
        # Concatenate all features along the last dimension
        combined_features = torch.cat([
            site_e,
            time_e,
            quality_e,
            driver_e,
            value_e
        ], dim=-1)  # Shape: [B, seq_len, embed_dim * 5]
        # Final projection to the model's hidden dimension
        final_embedding = self.final_projection(combined_features)  # Shape: [B, seq_len, hidden_dim]
        return final_embedding
