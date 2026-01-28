import torch
import torch.nn as nn


class CNN_LSTM(nn.Module):
    def __init__ ( self,args):
        """
        修改后的CNN-LSTM模型 (v2_Revised)。

        核心改动:
        1. 移除了 Conv2d 中的 'padding'，使数据流(H)自然收缩。
        2. 明确计算了 'flattened_features' 的大小，不再使用虚拟输入。

        参数:
        input_features (int): 输入特征的数量 (例如 12)
        lstm_hidden_units (int): LSTM 层的隐藏单元数
        dropout_rate (float): Dropout 的比率
        """
        super(CNN_LSTM, self).__init__()
        self.input_features = args.feature_in
        self.hidden_dim = args.hidden_dim
        self.dropout_rate = args.dropout_rate
        self.output_dim = args.output_dim
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(2, 1))
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1), stride=(1, 1))

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 1))
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1), stride=(1, 1))

        self.dropout_cnn = nn.Dropout(p=self.dropout_rate)
        h_final = self.input_features - 4
        # 检查：确保我们的网络有足够的深度
        if h_final <= 0:
            raise ValueError('feature error')
        self.flattened_features = 32 * h_final * 1
        # --- LSTM 块 ---
        self.lstm = nn.LSTM(
            input_size=self.flattened_features,
            hidden_size=self.hidden_dim,
            batch_first=True,  # 接受 (N, T, Features)
            num_layers=2,
        )

        # --- 全连接 (FC) 块 ---
        self.dropout_fc = nn.Dropout(p=self.dropout_rate)
        # 因为是双向 (bidirectional=True)，所以输入是 hidden_units * 2
        self.fc = nn.Linear(self.hidden_dim,self.output_dim )

    def forward ( self, x ):
        """
        x 的期望输入形状: (N, T, F) = (batch_size, sequence_length, input_features)
        """
        x_his = x['driver_history']
        x_cur = x['driver_current']
        x_cur = x_cur.unsqueeze(1)
        x = torch.cat([x_his[:,:,0:self.input_features], x_cur[:,:,0:self.input_features]], dim=1)
        batch_size, seq_len, features = x.shape
        # 1. Reshape 以便 CNN 处理: (N, T, F) -> (N*T, 1, F, 1)
        x_cnn = x.view(-1, 1, self.input_features, 1)
        # 2. 通过 CNN 块:
        # (N*T, 1, H=12, 1) -> Conv1 -> (N*T, 16, H=11, 1)
        c = self.relu(self.bn1(self.conv1(x_cnn)))
        # (N*T, 16, H=11, 1) -> Pool1 -> (N*T, 16, H=10, 1)
        c = self.pool1(c)
        # (N*T, 16, H=10, 1) -> Conv2 -> (N*T, 32, H=9, 1)
        c = self.relu(self.bn2(self.conv2(c)))
        # (N*T, 32, H=9, 1) -> Pool2 -> (N*T, 32, H=8, 1)
        c = self.pool2(c)
        c = self.dropout_cnn(c)
        # 3. 展平 CNN 输出: (N*T, 32, H=8, 1) -> (N*T, 32*8*1 = 256)
        c_flat = torch.flatten(c, 1)  # 从维度 1 (通道) 开始展平
        # 4. Reshape 以便 LSTM 处理: (N*T, 256) -> (N, T, 256)
        r_in = c_flat.view(batch_size, seq_len, self.flattened_features)
        # 5. 通过 LSTM: (N, T, 256) -> (N, T, H*D = 128)
        r_out, (h_n, c_n) = self.lstm(r_in)
        # 6. 获取最后一个时间步的输出: (N, T, 128) -> (N, 128)
        last_hidden_state = r_out[:, -1, :]
        # 7. 通过 FC 块: (N, 128) -> (N, 1)
        output = self.dropout_fc(last_hidden_state)
        output = self.fc(output)
        return output


