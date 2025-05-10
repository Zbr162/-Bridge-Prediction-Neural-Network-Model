import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import math
import os
import warnings

warnings.filterwarnings("ignore")

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['STSong']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

# 检查CUDA是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


# 自定义数据集
class SeismicDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets).reshape(-1, 1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


# TimesNet核心组件: 时频转换模块 - 修复版
class TimeFrequencyTransform(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=3):
        super(TimeFrequencyTransform, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 2D卷积处理频域信息
        self.freq_conv = nn.Sequential(
            nn.Conv2d(1, hidden_size, kernel_size=(3, kernel_size), padding=(1, kernel_size // 2)),
            nn.BatchNorm2d(hidden_size),
            nn.GELU(),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=(3, kernel_size), padding=(1, kernel_size // 2)),
            nn.BatchNorm2d(hidden_size),
            nn.GELU()
        )

        # 输出映射
        self.output_proj = nn.Conv2d(hidden_size, 1, kernel_size=(1, 1))

    def forward(self, x):
        # x: [Batch, Length, Channel]
        batch_size, seq_len, _ = x.shape

        # 使用FFT变换到频域 - 添加小值避免数值问题
        x_fft = torch.fft.rfft(x + 1e-8, dim=1)
        x_freq = torch.stack([x_fft.real, x_fft.imag], dim=1)  # [B, 2, L/2+1, C]

        # 压缩变成单通道表示
        x_freq = x_freq.mean(dim=1, keepdim=True)  # [B, 1, L/2+1, C]

        # 2D卷积处理频域信息
        freq_out = self.freq_conv(x_freq)  # [B, H, L/2+1, C]
        freq_out = self.output_proj(freq_out)  # [B, 1, L/2+1, C]
        freq_out = freq_out.squeeze(1)  # [B, L/2+1, C]

        # 限制极端值，增加数值稳定性
        freq_out = torch.clamp(freq_out, -1e5, 1e5)

        # 确保输出与输入维度匹配
        # 将频域特征转回时域 (逆FFT)
        freq_out_complex = torch.complex(freq_out, torch.zeros_like(freq_out))
        time_out = torch.fft.irfft(freq_out_complex, n=seq_len, dim=1)  # [B, L, C]

        return time_out


# 周期化嵌入模块 (TimesNet的关键创新) - 修复版
class PeriodicalEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PeriodicalEmbedding, self).__init__()
        self.period_detector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 4)  # 探测4种可能的周期
        )

        self.embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

    def forward(self, x):
        # x: [B, L, C]
        batch, length, _ = x.shape

        # 探测最可能的周期长度 - 使用全局平均池化来减少特征维度
        x_mean = torch.mean(x, dim=1)  # [B, C]
        period_weight = torch.softmax(self.period_detector(x_mean), dim=-1)  # [B, 4]

        # 使用不同周期长度重塑序列
        periods = [2, 4, 8, 16]  # 可能的周期长度
        period_features = []

        for i, p in enumerate(periods):
            if length % p != 0:
                pad_length = p - (length % p)
                x_pad = torch.cat([x, torch.zeros(batch, pad_length, x.size(2)).to(x.device)], dim=1)
                reshape_length = x_pad.size(1)
            else:
                x_pad = x
                reshape_length = length

            # 重塑为[batch, period, reshape_length/period, channels]
            x_reshaped = x_pad.reshape(batch, p, reshape_length // p, -1)

            # 对周期维度进行2D卷积，但先交换维度
            x_period = x_reshaped.permute(0, 3, 1, 2)  # [B, C, p, reshape_length/p]

            # 简单的平均池化以提取周期特征
            x_period = torch.mean(x_period, dim=-1)  # [B, C, p]
            x_period = x_period.permute(0, 2, 1)  # [B, p, C]

            # 重复扩展到原始长度
            x_period = x_period.unsqueeze(2).expand(-1, -1, reshape_length // p, -1)  # [B, p, reshape_length/p, C]
            x_period = x_period.reshape(batch, reshape_length, -1)[:, :length, :]  # [B, L, C]

            # 避免极端值
            x_period = torch.clamp(x_period, -1e5, 1e5)
            period_features.append(x_period * period_weight[:, i].view(-1, 1, 1))

        # 合并周期特征
        period_embedding = torch.sum(torch.stack(period_features), dim=0)

        # 限制值范围
        period_embedding = torch.clamp(period_embedding, -1e5, 1e5)

        # 映射到隐藏维度
        return self.embed(period_embedding)


# 增强型多头自注意力 - 修复版
class EnhancedMultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super(EnhancedMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)

        # 多头注意力的线性投影
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        # 相对位置编码
        self.max_seq_len = 1024
        self.rel_pos_embedding = nn.Parameter(torch.zeros(2 * self.max_seq_len - 1, hidden_size // num_heads))
        nn.init.xavier_uniform_(self.rel_pos_embedding)

    def forward(self, x):
        # x: [batch, seq_len, hidden_size]
        batch_size, seq_len, _ = x.shape

        # 多头投影
        q = self.query(x).view(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 1, 3)  # [B, H, L, D/H]
        k = self.key(x).view(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 1, 3)  # [B, H, L, D/H]
        v = self.value(x).view(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 1, 3)  # [B, H, L, D/H]

        # 计算注意力得分，添加数值稳定性
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.hidden_size // self.num_heads)  # [B, H, L, L]

        # 限制极端值
        scores = torch.clamp(scores, -1e5, 1e5)

        # 添加相对位置编码 (简化实现)
        if seq_len <= self.max_seq_len:
            pos_indices = torch.arange(seq_len).unsqueeze(1) - torch.arange(seq_len).unsqueeze(0) + self.max_seq_len - 1
            pos_indices = pos_indices.to(x.device)
            rel_pos = self.rel_pos_embedding[pos_indices].unsqueeze(0).unsqueeze(0)  # [1, 1, L, L, D/H]
            dim_head = self.hidden_size // self.num_heads
            rel_pos_score = torch.matmul(q.unsqueeze(-2), rel_pos.transpose(-1, -2)).squeeze(-2)  # [B, H, L, L]
            scores = scores + rel_pos_score

        # 注意力权重 - 添加数值稳定性
        # 使用带有掩码的softmax以增加稳定性
        attn_mask = torch.ones(scores.size(), device=scores.device)
        attn = torch.nn.functional.softmax(scores, dim=-1) * attn_mask
        # 归一化掩码后的注意力权重
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
        attn = self.dropout(attn)

        # 加权上下文向量
        context = torch.matmul(attn, v)  # [B, H, L, D/H]
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)  # [B, L, D]

        # 输出投影
        output = self.out_proj(context)  # [B, L, D]

        return output


# 时空傅立叶自注意力模块 - 修复版
class TimeFrequencyAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super(TimeFrequencyAttention, self).__init__()
        self.hidden_size = hidden_size

        # 时域注意力
        self.time_attn = EnhancedMultiHeadAttention(hidden_size, num_heads, dropout)

        # 频域变换
        self.fft_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU()
        )

        # 频域注意力
        self.freq_attn = EnhancedMultiHeadAttention(hidden_size, num_heads, dropout)

        # 输出融合
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )

    def forward(self, x):
        # 时域路径
        time_out = self.time_attn(x)

        # 频域路径 - FFT变换 - 添加小值避免数值问题
        x_fft = torch.fft.rfft(x + 1e-8, dim=1)
        x_fft_real = x_fft.real
        x_fft_imag = x_fft.imag

        # 频域特征变换 - 使用复数的模 - 添加小值避免平方根为零
        x_fft_amp = torch.sqrt(x_fft_real ** 2 + x_fft_imag ** 2 + 1e-10)
        x_fft_transformed = self.fft_layer(x_fft_amp)

        # 频域注意力
        freq_out = self.freq_attn(x_fft_transformed)

        # 限制极端值
        freq_out = torch.clamp(freq_out, -1e5, 1e5)

        # 逆FFT将频域特征转回时域
        freq_out_time = torch.fft.irfft(
            torch.complex(freq_out, torch.zeros_like(freq_out)),
            n=x.size(1),
            dim=1
        )

        # 融合时域和频域特征
        combined = torch.cat([time_out, freq_out_time], dim=-1)
        output = self.output_layer(combined)

        # 额外的限制确保输出稳定
        output = torch.clamp(output, -1e5, 1e5)

        return output


# 多尺度CNN特征提取器
class MultiScaleCNN(nn.Module):
    def __init__(self, input_channels=1, hidden_channels=64):
        super(MultiScaleCNN, self).__init__()

        # 多尺度卷积块
        self.conv_small = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.GELU(),
            nn.Conv1d(hidden_channels // 2, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU()
        )

        self.conv_medium = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels // 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.GELU(),
            nn.Conv1d(hidden_channels // 2, hidden_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU()
        )

        self.conv_large = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels // 2, kernel_size=9, padding=4),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.GELU(),
            nn.Conv1d(hidden_channels // 2, hidden_channels, kernel_size=9, padding=4),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU()
        )

        # 注意力融合
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(hidden_channels * 3, hidden_channels * 3 // 8, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels * 3 // 8, hidden_channels * 3, kernel_size=1),
            nn.Sigmoid()
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Conv1d(hidden_channels * 3, hidden_channels, kernel_size=1),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU()
        )

    def forward(self, x):
        # x: [B, C, L]
        small_scale = self.conv_small(x)
        medium_scale = self.conv_medium(x)
        large_scale = self.conv_large(x)

        # 特征融合
        features = torch.cat([small_scale, medium_scale, large_scale], dim=1)

        # 注意力机制
        attention_weights = self.attention(features)
        features = features * attention_weights

        return self.fusion(features)  # [B, hidden_channels, L]


# 残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(ResidualBlock, self).__init__()
        padding = dilation * (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(0.2)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.GELU()

        # 如果输入输出通道不一致，添加1x1卷积进行调整
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.act2(out)

        return out


# 时间卷积网络(TCN)
class TCNBlock(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=3, dropout=0.2):
        super(TCNBlock, self).__init__()

        layers = []
        num_levels = 4
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = input_size if i == 0 else hidden_size
            out_channels = hidden_size

            layers.append(ResidualBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation
            ))

        self.network = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.network(x))


# 组合CNN+TCN+LSTM+TimesNet模型
class TimesNetEnhancedModel(nn.Module):
    def __init__(self, seq_len=48, hidden_size=64, lstm_layers=2, dropout=0.2):
        super(TimesNetEnhancedModel, self).__init__()

        # 1. 多尺度CNN特征提取
        self.cnn = MultiScaleCNN(input_channels=1, hidden_channels=hidden_size)

        # 2. TimesNet核心组件: 时频变换
        self.time_freq_transform = TimeFrequencyTransform(
            input_size=seq_len,
            hidden_size=hidden_size
        )

        # 3. 周期性嵌入
        self.period_embedding = PeriodicalEmbedding(
            input_dim=hidden_size,
            hidden_dim=hidden_size
        )

        # 4. TCN层处理时序信息
        self.tcn = TCNBlock(
            input_size=hidden_size,
            hidden_size=hidden_size,
            kernel_size=3,
            dropout=dropout
        )

        # 5. 时频自注意力
        self.tf_attention = TimeFrequencyAttention(
            hidden_size=hidden_size,
            num_heads=4,
            dropout=dropout
        )

        # 6. LSTM层
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        # 7. 全局上下文编码
        self.global_context = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 8. 特征整合
        total_features = hidden_size * 2  # 双向LSTM
        total_features += hidden_size  # 周期性嵌入
        total_features += hidden_size  # 时频注意力

        # 9. 预测头
        self.prediction_head = nn.Sequential(
            nn.Linear(total_features, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        # x: [B, L, 1]
        batch_size, seq_len, _ = x.shape

        # 1. CNN特征提取
        x_cnn = x.transpose(1, 2)  # [B, 1, L]
        cnn_features = self.cnn(x_cnn)  # [B, H, L]

        # 将CNN特征转回[B, L, H]格式
        cnn_features = cnn_features.transpose(1, 2)  # [B, L, H]

        # 2. TimesNet时频变换
        time_freq_features = self.time_freq_transform(cnn_features)

        # 3. 周期性嵌入
        period_features = self.period_embedding(cnn_features)

        # 特征增强: 添加CNN特征、时频特征和周期性特征
        # 添加裁剪以确保数值稳定性
        enhanced_features = torch.clamp(cnn_features + time_freq_features + period_features, -1e5, 1e5)

        # 4. 时频自注意力
        tf_features = self.tf_attention(enhanced_features)

        # 5. TCN处理
        # 转换格式为[B, C, L]
        tcn_input = enhanced_features.transpose(1, 2)
        tcn_features = self.tcn(tcn_input)  # [B, H, L]
        tcn_features = tcn_features.transpose(1, 2)  # [B, L, H]

        # 6. LSTM处理 - 添加异常值检测和处理
        combined_features = torch.clamp(tcn_features + tf_features, -1e5, 1e5)
        lstm_out, (h_n, _) = self.lstm(combined_features)  # lstm_out: [B, L, H*2]

        # 7. 获取LSTM最后隐状态
        last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)  # [B, H*2]

        # 8. 特征融合
        # 获取序列末尾特征
        period_out = period_features[:, -1, :]  # [B, H]
        tf_out = tf_features[:, -1, :]  # [B, H]

        # 组合时域和频域特征
        combined = torch.cat([
            last_hidden,  # LSTM特征 [B, H*2]
            period_out,  # 周期性特征 [B, H]
            tf_out  # 时频特征 [B, H]
        ], dim=1)

        # 确保稳定性
        combined = torch.clamp(combined, -1e5, 1e5)

        # 9. 预测
        return self.prediction_head(combined)


# 权重初始化函数，提高模型数值稳定性
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight, gain=0.5)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)


# 数据预处理函数 - 经过增强的版本
def load_and_preprocess_data(features_path, target_path, window_size=48, stride=12):
    """增强的数据预处理函数，添加异常值处理"""
    print("加载数据...")
    features = pd.read_csv(features_path, header=None).T
    target = pd.read_csv(target_path, header=None).T

    print(f"原始数据形状 - 特征: {features.shape}, 目标: {target.shape}")

    print("预处理数据...")
    # 检测和处理异常值
    features_np = features.values
    target_np = target.values

    # 裁剪极端值以避免数值问题
    features_np = np.clip(features_np, -10, 10)  # 根据实际数据分布调整范围
    target_np = np.clip(target_np, -10, 10)

    # 标准化比归一化更适合时间序列数据
    scaler_features = StandardScaler()
    scaler_target = StandardScaler()

    features_scaled = scaler_features.fit_transform(features_np)
    target_scaled = scaler_target.fit_transform(target_np)

    # 处理NaN值
    features_scaled = np.nan_to_num(features_scaled)
    target_scaled = np.nan_to_num(target_scaled)

    print("创建滑动窗口...")
    X, y = [], []
    total_sequences = len(features_scaled)

    for i in tqdm(range(total_sequences), desc="处理序列"):
        sequence_length = len(features_scaled[i])
        for j in range(0, sequence_length - window_size - 1, stride):
            X.append(features_scaled[i, j:j + window_size])
            y.append(target_scaled[i, j + window_size])

    X = np.array(X).reshape(-1, window_size, 1)
    y = np.array(y).reshape(-1, 1)

    print(f"最终数据形状 - X: {X.shape}, y: {y.shape}")
    return X, y, scaler_features, scaler_target


# 修改训练函数以增强数值稳定性
def train_model(model, train_loader, val_loader, num_epochs, device, patience=15):
    """增强的训练函数，添加了NaN处理和更稳定的梯度裁剪"""
    criterion = nn.MSELoss()
    # 降低初始学习率
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)

    # 余弦退火学习率调度
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # 第一次重启的轮次
        T_mult=2,  # 每次重启后周期长度倍增
        eta_min=1e-6  # 最小学习率
    )

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    no_improve = 0
    nan_count = 0  # 跟踪NaN出现次数

    epoch_pbar = trange(num_epochs, desc="训练中")

    for epoch in epoch_pbar:
        # 训练阶段
        model.train()
        train_loss = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [训练]", leave=False)
        for batch_x, batch_y in train_pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()

            # 使用try/except捕获可能的运行时错误
            try:
                outputs = model(batch_x)

                # 检查输出是否包含NaN
                if torch.isnan(outputs).any():
                    print("警告: 模型输出包含NaN。跳过此批次...")
                    nan_count += 1
                    if nan_count > 5:  # 如果连续出现5次NaN，降低学习率
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= 0.5
                        print(f"检测到过多NaN，降低学习率至 {optimizer.param_groups[0]['lr']}")
                        nan_count = 0
                    continue

                loss = criterion(outputs, batch_y)

                # 检查损失是否为NaN
                if torch.isnan(loss):
                    print("警告: 损失为NaN。跳过此批次...")
                    continue

                loss.backward()

                # 检查梯度是否包含NaN或Inf
                has_nan_or_inf = False
                for param in model.parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            has_nan_or_inf = True
                            break

                if has_nan_or_inf:
                    print("警告: 梯度包含NaN或Inf。跳过此更新...")
                    continue

                # 更激进的梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

                optimizer.step()

                train_loss += loss.item()
                train_pbar.set_postfix({'损失': f'{loss.item():.6f}'})

            except RuntimeError as e:
                print(f"运行时错误: {e}")
                continue

        # 更新学习率
        scheduler.step()

        # 验证阶段
        model.eval()
        val_loss = 0

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [验证]", leave=False)
        with torch.no_grad():
            for batch_x, batch_y in val_pbar:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                try:
                    outputs = model(batch_x)

                    # 检查输出是否包含NaN
                    if torch.isnan(outputs).any():
                        print("警告: 验证中模型输出包含NaN。")
                        continue

                    val_batch_loss = criterion(outputs, batch_y).item()
                    val_loss += val_batch_loss

                    val_pbar.set_postfix({'损失': f'{val_batch_loss:.6f}'})
                except RuntimeError:
                    continue

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        current_lr = optimizer.param_groups[0]['lr']

        epoch_pbar.set_postfix({
            '训练损失': f'{train_loss:.6f}',
            '验证损失': f'{val_loss:.6f}',
            '学习率': f'{current_lr:.7f}'
        })

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"\n保存最佳模型，验证损失: {val_loss:.6f}")
            torch.save(model.state_dict(), 'best_timesnet_model.pth')
            no_improve = 0
        else:
            no_improve += 1

        # 早停
        if no_improve >= patience:
            print(f"\n{epoch + 1}轮后触发早停！")
            break

    return train_losses, val_losses


# 评估函数
def evaluate_model(model, test_loader, scaler_target, device):
    """模型评估函数"""
    model.eval()
    predictions = []
    actuals = []

    # 使用tqdm显示评估进度
    test_pbar = tqdm(test_loader, desc="评估中")
    with torch.no_grad():
        for batch_x, batch_y in test_pbar:
            batch_x = batch_x.to(device)
            try:
                outputs = model(batch_x)
                # 检查输出是否包含NaN
                if torch.isnan(outputs).any():
                    continue
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(batch_y.numpy())
            except RuntimeError:
                continue

    # 转换为numpy数组
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # 创建一个新的StandardScaler，只用于单个特征
    new_scaler = StandardScaler()
    # 使用原始scaler的参数来设置新scaler
    new_scaler.mean_ = np.array([scaler_target.mean_[0]])
    new_scaler.scale_ = np.array([scaler_target.scale_[0]])
    new_scaler.var_ = np.array([scaler_target.var_[0]])
    new_scaler.n_features_in_ = 1

    # 反标准化
    predictions = new_scaler.inverse_transform(predictions)
    actuals = new_scaler.inverse_transform(actuals)

    # 确保是一维数组
    predictions = predictions.ravel()
    actuals = actuals.ravel()

    # 计算评估指标
    r2 = r2_score(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)

    return {
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae,
        'predictions': predictions,
        'actuals': actuals
    }


# 可视化函数
def plot_results(results, train_losses, val_losses):
    """结果可视化函数"""
    # 创建一个更大的图形
    plt.figure(figsize=(20, 10))

    # 预测对比图 - 前500个样本
    plt.subplot(2, 2, 1)
    plt.plot(results['actuals'][:500], label='真实值', alpha=0.7)
    plt.plot(results['predictions'][:500], label='预测值', alpha=0.7)
    plt.legend()
    plt.title(f'预测结果对比 (前500个样本)\nR² = {results["R2"]:.4f}')
    plt.xlabel('样本索引')
    plt.ylabel('值')

    # 预测对比图 - 中间500个样本
    middle_idx = len(results['actuals']) // 2
    plt.subplot(2, 2, 2)
    plt.plot(results['actuals'][middle_idx:middle_idx + 500], label='真实值', alpha=0.7)
    plt.plot(results['predictions'][middle_idx:middle_idx + 500], label='预测值', alpha=0.7)
    plt.legend()
    plt.title(f'预测结果对比 (中间500个样本)\nR² = {results["R2"]:.4f}')
    plt.xlabel('样本索引')
    plt.ylabel('值')

    # 损失曲线
    plt.subplot(2, 2, 3)
    plt.plot(train_losses, label='训练损失', alpha=0.7)
    plt.plot(val_losses, label='验证损失', alpha=0.7)
    plt.legend()
    plt.title('训练过程损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # 真实值vs预测值的散点图
    plt.subplot(2, 2, 4)
    plt.scatter(results['actuals'], results['predictions'], alpha=0.1, s=1)
    plt.plot([min(results['actuals']), max(results['actuals'])],
             [min(results['actuals']), max(results['actuals'])],
             'r--', label='理想预测线')
    plt.legend()
    plt.title(f'真实值 vs 预测值\nR² = {results["R2"]:.4f}')
    plt.xlabel('真实值')
    plt.ylabel('预测值')

    plt.tight_layout()
    plt.savefig('timesnet_enhanced_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 额外创建一个图显示最后500个样本
    plt.figure(figsize=(12, 6))
    plt.plot(results['actuals'][-500:], label='真实值', alpha=0.7)
    plt.plot(results['predictions'][-500:], label='预测值', alpha=0.7)
    plt.legend()
    plt.title(f'预测结果对比 (最后500个样本)\nR² = {results["R2"]:.4f}')
    plt.xlabel('样本索引')
    plt.ylabel('值')
    plt.tight_layout()
    plt.savefig('timesnet_enhanced_last_samples.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    # 参数设置
    WINDOW_SIZE = 48  # 输入窗口大小
    STRIDE = 12  # 滑动步长
    BATCH_SIZE = 128  # 批次大小
    NUM_EPOCHS = 150  # 最大训练轮数

    print("开始数据预处理...")
    X, y, scaler_features, scaler_target = load_and_preprocess_data(
        'xtrain_new.csv',
        'ytrain_new.csv',
        WINDOW_SIZE,
        STRIDE
    )

    print("划分数据集...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, shuffle=False
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, shuffle=False
    )

    print("创建数据加载器...")
    train_dataset = SeismicDataset(X_train, y_train)
    val_dataset = SeismicDataset(X_val, y_val)
    test_dataset = SeismicDataset(X_test, y_test)

    # 根据系统设置worker数
    num_workers = 0  # 如果有多核CPU，可以设置为4或更高

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=num_workers)

    print("初始化TimesNet增强模型...")
    model = TimesNetEnhancedModel(
        seq_len=WINDOW_SIZE,
        hidden_size=64,
        lstm_layers=2,
        dropout=0.25
    ).to(device)

    # 应用权重初始化提高稳定性
    model.apply(init_weights)

    # 打印模型结构与参数量
    print("\n模型架构:")
    print(model)
    print(f"\n总参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练模型
    print("\n开始训练...")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, NUM_EPOCHS, device, patience=15
    )

    print("加载最佳模型...")
    model.load_state_dict(torch.load('best_timesnet_model.pth'))

    # 评估模型
    print("评估模型...")
    results = evaluate_model(model, test_loader, scaler_target, device)

    print("\n=== 模型评估结果 ===")
    print(f"R2 分数: {results['R2']:.4f}")
    print(f"RMSE: {results['RMSE']:.4f}")
    print(f"MAE: {results['MAE']:.4f}")

    # 绘制结果
    print("\n绘制结果...")
    plot_results(results, train_losses, val_losses)

    print("\n训练完成!")


if __name__ == '__main__':
    main()
