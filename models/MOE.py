import torch
import torch.nn as nn
import torch.nn.functional as F
from config.cfg import get_config
from .RMSNorm import RMSNorm


class Gate(nn.Module):
    # Reference deepseek v3's innovative load balancing strategy without additional loss
    def __init__ ( self, config ):
        super().__init__()
        # Basic linear transformation
        self.base_gate = nn.Linear(config.hidden_dim, config.expert_num)
        # Learnable expert bias term
        self.bias = nn.Parameter(torch.zeros(config.expert_num))

    def forward ( self, x ):
        # x shape is [token_num, embed_dim * 4]
        # base_scores shape is [token_num, expert_num]
        base_scores = self.base_gate(x)
        # Add dynamic bias for each expert
        biased_scores = base_scores + self.bias.unsqueeze(0)
        return biased_scores


class Expert_MLP(nn.Module):
    def __init__ ( self, hidden_dim, dropout_rate ):
        """
        MLP expert module
        Parameters:
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout_rate)
        )
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)

    def forward ( self, x ):
        """
        Parameters:
            x: Tensor after embedding, shape (token,embed_dim*4)
        Returns:
            Output tensor, shape (token,hidden_dim)
        """
        res = x
        out = self.norm(x)
        # MLP processing
        mlp_out = self.mlp(out)
        # Apply layer normalization
        return mlp_out + res


class MOERouter(nn.Module):
    def __init__ ( self, config ):
        super().__init__()
        self.gate = Gate(config)
        self.expert_num = config.expert_num
        self.top_k = config.top_k

    def forward ( self, input ):
        # Calculate expert routing scores
        router_logits = self.gate(input)  # shape is (b * s, expert_number)
        # Calculate the probability of each expert after softmax
        # For each token for expert_number experts
        routing_probs = F.softmax(router_logits, dim=-1, dtype=torch.float)
        # Calculate the output of top_k experts, top_k is backpropagatable
        # Shape are (b * s, top_k), where router_weights are the weights of selected top_k experts, and selected_experts are the indices of selected experts
        router_weights, selected_experts = torch.topk(
            routing_probs, self.top_k, dim=-1
        )
        # This part is to make the probability of selected top_k experts sum to 1, normalize the probability
        router_weights = router_weights / router_weights.sum(dim=-1, keepdim=True)
        router_weights = router_weights.to(input.dtype)
        # Generate expert mask
        # Shape is (b * s, top_k, expert_number) , meaning for each token selected top_k experts, use 8-bit binary number encoding.
        expert_mask = F.one_hot(
            selected_experts,
            num_classes=self.expert_num
        )
        # For example, with two experts and top_k=2, token_num = 8: the first dimension represents the number of experts, the second dimension represents top_k, and the third dimension represents token_num. When expert[0,0,0]=1, it means the first expert is top_1 for the first token.
        expert_mask = expert_mask.permute(2, 1, 0)  # (expert_number, top_k, b * s)

        return router_logits, router_weights, selected_experts, expert_mask


class SparseMOE(nn.Module):

    # Sparse MOE model, here each token will pass through topk experts to get the hidden states corresponding to the token
    def __init__ ( self, config ):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.expert_num = config.expert_num
        self.experts = nn.ModuleList(
            [
                Expert_MLP(self.hidden_dim, dropout_rate=config.dropout_rate)
                for _ in range(self.expert_num)
            ]
        )
        self.router = MOERouter(config)

    def forward ( self, x ):
        # x shape is (b*s, embed_dim*5)
        token_num, x_embed_dim = x.size()
        # router_logits shape  (b * s, expert_number) is the value output by Gate, representing the score of each expert
        # router_weights shape is (b * s, top_k) is the weight of selected top_k experts (normalized)
        # selected_experts_indices shape is (b * s, top_k) is the index of selected experts for each token
        # expert_mask shape is (expert_number, top_k, b * s) is the mask of selected experts
        router_logits, router_weights, selected_experts_indices, expert_mask = self.router(x)
        # Initialize result array
        final_hidden_states = torch.zeros(
            (token_num, self.hidden_dim),
            dtype=x.dtype,
            device=x.device
        )  # Initialize a final vector first
        # Traverse each expert, calculate the hidden_state of the token selected by this expert, and then add it to
        # final_hidden_states
        for expert_idx in range(self.expert_num):
            # Get the current expert model
            expert_layer = self.experts[expert_idx]
            # expert_mask[expert_idx] shape is (top_k, b * s) represents the mask of the current expert
            # top_idx is obtained by torch.where() function, representing the row index where the mask of the current expert is 1. It means that the token selected by the current expert is the top several experts. For example, if top_k = 2, top_num_idx = 0 means the current expert is top1, top_num_idx = 1 means the current expert is top2
            # token_idx is obtained by torch.where() function, representing the column index where the mask of the current expert is 1. It represents the position index of the token selected by the current expert in batch*seq_len
            top_num_idx, token_idx = torch.where(expert_mask[expert_idx] == 1)
            # Get the token selected by the current expert
            # current_state shape：（selected_token_number, embed_dim*4）
            current_state = x.unsqueeze(
                0
            )[:, token_idx, :].reshape(-1, x_embed_dim)
            # router_weight's shape is (b * s, top_k)
            # Multiply the output of the current expert by the weight of the current expert to get the final output of the current expert
            current_hidden_states = expert_layer(
                current_state
            ) * router_weights[token_idx, top_num_idx].unsqueeze(
                -1)  # （selected_token_number, 1） There is broadcasting here

            # Add the output of the current expert to final_hidden_states, in-place operation
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(x.dtype))
        # final_hidden_states still shape (batch_size*seq_len, self.hidden_dim)
        return final_hidden_states


class DeepSeekMoE(nn.Module):
    def __init__ ( self, config ):
        super().__init__()
        self.shared_expert_num = config.shared_expert_num
        self.embed_dim = config.embed_dim
        self.hidden_dim = config.hidden_dim
        self.expert_num = config.expert_num
        self.top_k = config.top_k
        self.shared_experts = nn.ModuleList(
            [
                Expert_MLP(hidden_dim=self.hidden_dim,
                           dropout_rate=config.dropout_rate)
                for _ in
                range(self.shared_expert_num)
            ]
        )
        self.routed_moe = SparseMOE(config)

    def forward ( self, x ):
        # x shape is (b , s, embed_dim*4)
        batch_size, seq_len, x_embed_dim = x.size()
        # reshape x to  (b * s, embed_dim*4)
        input_token = x.view(-1, x_embed_dim)
        # Calculate the output of shared experts shared_expert_number[b*s,  hidden_dim]
        shared_expert_outputs_list = [shared_expert(input_token) for shared_expert in self.shared_experts]
        # Concatenate the output of shared experts together [token num * shared_expert_num, hidden_dim]
        shared_expert_outputs = torch.stack(shared_expert_outputs_list, dim=0)
        # Sum
        shared_expert_outputs = shared_expert_outputs.sum(dim=0, keepdim=False)
        # Reshape the output of shared experts to [b, s, hidden_dim]
        shared_expert_outputs = shared_expert_outputs.reshape(batch_size, seq_len, -1)
        # Calculate the output of routed experts
        routed_expert_outputs = self.routed_moe(input_token)
        # Reshape the output of routed experts to [b, s, hidden_dim]
        routed_expert_outputs = routed_expert_outputs.reshape(batch_size, seq_len, -1)
        # Add the output of shared experts and routed experts
        return shared_expert_outputs + routed_expert_outputs


class VariableAttention(nn.Module):
    """
    为每个时间步，计算其在变量/特征维度上的注意力分数。
    这个模块的目标是识别出在某个特定时间点，哪些特征维度更重要。
    """

    def __init__ ( self, hidden_dim: int ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward ( self, x: torch.Tensor ):
        """
        Args:
            x (torch.Tensor): 输入张量，形状为 [B, seq_len, hidden_dim]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - weighted_x: 加权后的张量，形状与输入相同
                - attn_weights: 变量注意力权重，形状与输入相同
        """
        # 1. 计算每个特征维度的注意力分数
        # attn_scores shape: [B, seq_len, hidden_dim]
        attn_scores = self.mlp(x)
        # 2. 使用 softmax 将分数归一化为权重
        # 在最后一个维度（hidden_dim）上进行softmax
        # attn_weights shape: [B, seq_len, hidden_dim]
        attn_weights = self.softmax(attn_scores)
        # 3. 将权重应用到原始输入上（逐元素相乘）
        weighted_x = x * attn_weights
        return weighted_x, attn_weights


class DualAttentionLayer(nn.Module):

    def __init__ ( self, config ):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.variable_attention = VariableAttention(self.hidden_dim)
        self.var_attn_norm = RMSNorm(self.hidden_dim)
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=config.n_head,
            batch_first=True,
        )
        self.temp_norm = RMSNorm(self.hidden_dim)

    def forward ( self, x: torch.Tensor ):
        """
        Args:
            x (torch.Tensor): 输入张量，形状为 [B, seq_len, hidden_dim]

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - x_attended: 经过双重注意力处理后的输出张量
                - var_attn_weights: 变量注意力的权重
                - temporal_attn_weights: 时间注意力的权重
        """
        res_var = x
        x_weighted, var_attn_weights = self.variable_attention(x)
        # 应用残差连接和归一化
        x = self.var_attn_norm(res_var + x_weighted)
        temp_x = x
        x_attended, _ = self.temporal_attention(query=x, key=x, value=x, average_attn_weights=False)
        x = self.temp_norm(x_attended + temp_x)
        return x


class MoETransformerBlock(nn.Module):
    def __init__ ( self, config ):
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_dim)
        self.self_attn = DualAttentionLayer(config)
        self.moe = DeepSeekMoE(config)
        self.norm2 = RMSNorm(config.hidden_dim)

    def forward ( self, x ):
        res = x
        # Self-Attention部分：Pre-Norm
        x_norm = self.norm1(x)
        attn_out = self.self_attn(x_norm)
        x = res + attn_out  # 残差连接：原始输入 + attention输出
        # MoE部分：Pre-Norm
        res = x  # 保存当前状态作为残差
        x_norm = self.norm2(x)  # MoE前的Norm
        moe_out = self.moe(x_norm)
        x = res + moe_out  # 残差连接：attention输出 + MoE输出
        return x


class MoETransformer(nn.Module):
    def __init__ ( self, config ):
        super().__init__()
        self.layers = nn.ModuleList([
            MoETransformerBlock(config)
            for _ in range(config.n_layers)
        ])

    def forward ( self, x ):
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerBlock(nn.Module):
    def __init__ ( self, config ):
        super().__init__()
        self.self_attn = DualAttentionLayer(config)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, 4 * config.hidden_dim),
            nn.GELU(),
            nn.Linear(4 * config.hidden_dim, config.hidden_dim)
        )
        self.norm1 = RMSNorm(config.hidden_dim)
        self.norm2 = RMSNorm(config.hidden_dim)

    def forward ( self, x ):
        res = x
        x = self.norm1(x)
        attn_out = self.self_attn(x)
        x = self.norm2(res + attn_out)
        res = x
        ffn_out = self.ffn(x)
        return res + ffn_out


class Transformer_base(nn.Module):
    def __init__ ( self, config ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.n_layers)
        ])

    def forward ( self, x ):
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == '__main__':
    config = get_config()
    x = torch.randn(config.batch_size, config.seq_len, config.hidden_dim)
    model = Transformer_base(config)
    print(model(x))  # [batch_size,seq_len, hidden_dim]
