import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import os
import warnings
import scipy.stats
from scipy import signal

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


# 优化的CNN特征提取器 - 简化结构但保留多尺度能力
class OptimizedCNNExtractor(nn.Module):
    def __init__(self, input_channels=1, hidden_channels=128):
        super(OptimizedCNNExtractor, self).__init__()

        # 多尺度卷积块 - 简化版本
        self.conv_small = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
        )

        self.conv_medium = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Conv1d(hidden_channels * 2, hidden_channels, kernel_size=1),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU()
        )

        # 简化的注意力机制
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, L]
        small_scale = self.conv_small(x)
        medium_scale = self.conv_medium(x)

        # 特征融合
        features = torch.cat([small_scale, medium_scale], dim=1)
        features = self.fusion(features)

        # 简化的注意力机制
        attention_weights = self.attention(features)
        features = features * attention_weights

        return features  # [B, hidden_channels, L]


# 优化的残差块
class OptimizedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(OptimizedResidualBlock, self).__init__()
        padding = dilation * (kernel_size - 1) // 2

        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels)
        )

        # 如果输入输出通道不一致，添加1x1卷积进行调整
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels)
            )

        self.activation = nn.GELU()

    def forward(self, x):
        residual = x
        out = self.conv_block(x)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return self.activation(out)


# 优化的TCN模块
class OptimizedTCN(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=5, dropout=0.2):
        super(OptimizedTCN, self).__init__()

        layers = []
        num_levels = 3  # 减少层数以简化模型
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = input_size if i == 0 else hidden_size
            out_channels = hidden_size

            layers.append(OptimizedResidualBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation
            ))

        self.network = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.network(x))


# 优化的注意力机制
class OptimizedAttention(nn.Module):
    def __init__(self, hidden_size):
        super(OptimizedAttention, self).__init__()
        self.query = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: [batch, seq_len, hidden]
        scores = self.query(x)  # [batch, seq_len, 1]
        weights = torch.softmax(scores, dim=1)  # [batch, seq_len, 1]

        # 加权求和
        context = torch.sum(x * weights, dim=1)  # [batch, hidden]
        return context, weights


# 优化后的组合模型
class OptimizedCombinedModel(nn.Module):
    def __init__(self, seq_len=64, hidden_size=128, lstm_layers=2, dropout=0.2):
        super(OptimizedCombinedModel, self).__init__()

        # 1. 优化的CNN特征提取
        self.cnn = OptimizedCNNExtractor(input_channels=1, hidden_channels=hidden_size)

        # 2. 优化的TCN层
        self.tcn = OptimizedTCN(
            input_size=hidden_size,
            hidden_size=hidden_size,
            kernel_size=5,
            dropout=dropout
        )

        # 3. 双向LSTM层
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        # 4. 注意力层
        self.attention = OptimizedAttention(hidden_size * 2)

        # 5. 频域特征提取
        self.freq_feature = nn.Sequential(
            nn.Linear(seq_len // 2 + 1, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 6. 特征融合 - 简化版本
        total_features = hidden_size * 2  # LSTM特征
        total_features += hidden_size  # 频域特征

        # 7. 预测头 - 简化版本
        self.prediction_head = nn.Sequential(
            nn.Linear(total_features, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        # x: [B, L, 1]
        batch_size, seq_len, _ = x.shape

        # 提取频域特征
        x_flat = x.reshape(batch_size, seq_len)
        x_fft = torch.abs(torch.fft.rfft(x_flat, dim=1))
        freq_features = self.freq_feature(x_fft)

        # CNN特征提取
        x_cnn = x.transpose(1, 2)  # [B, 1, L]
        cnn_features = self.cnn(x_cnn)  # [B, hidden_size, L]

        # TCN处理
        tcn_features = self.tcn(cnn_features)  # [B, hidden_size, L]

        # LSTM处理
        lstm_input = tcn_features.transpose(1, 2)  # [B, L, hidden_size]
        lstm_out, _ = self.lstm(lstm_input)  # [B, L, hidden_size*2]

        # 注意力处理
        context, _ = self.attention(lstm_out)  # [B, hidden_size*2]

        # 组合特征
        combined = torch.cat([
            context,  # [B, hidden_size*2]
            freq_features  # [B, hidden_size]
        ], dim=1)

        # 预测
        return self.prediction_head(combined)


# 优化的平衡损失函数
class BalancedLoss(nn.Module):
    def __init__(self, mse_weight=0.6, mae_weight=0.3, peak_weight=0.1, peak_threshold=0.75):
        super(BalancedLoss, self).__init__()
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self.peak_weight = peak_weight
        self.peak_threshold = peak_threshold
        self.mse = nn.MSELoss(reduction='none')
        self.mae = nn.L1Loss(reduction='none')

    def forward(self, y_pred, y_true):
        # 基础损失
        mse_loss = torch.mean(self.mse(y_pred, y_true))
        mae_loss = torch.mean(self.mae(y_pred, y_true))

        # 峰值损失（更温和的权重）
        with torch.no_grad():
            batch_max = torch.max(torch.abs(y_true))
            peak_mask = (torch.abs(y_true) > self.peak_threshold * batch_max).float()

        # 只对峰值区域计算额外MSE
        peak_losses = self.mse(y_pred, y_true) * peak_mask
        peak_loss = torch.sum(peak_losses) / (torch.sum(peak_mask) + 1e-8)

        # 组合损失
        return self.mse_weight * mse_loss + self.mae_weight * mae_loss + self.peak_weight * peak_loss


# 优化的数据预处理函数
def optimized_load_and_preprocess_data(features_path, target_path, window_size=64, stride=8):
    """优化的数据预处理函数"""
    print("加载数据...")
    features = pd.read_csv(features_path, header=None).T
    target = pd.read_csv(target_path, header=None).T

    print(f"原始数据形状 - 特征: {features.shape}, 目标: {target.shape}")

    print("预处理数据...")
    # 使用RobustScaler处理异常值
    scaler_features = RobustScaler()
    scaler_target = RobustScaler()

    features_scaled = scaler_features.fit_transform(features)
    target_scaled = scaler_target.fit_transform(target)

    print("创建滑动窗口...")
    X, y = [], []
    total_sequences = features_scaled.shape[0]

    # 使用更小的stride生成更多样本
    for i in tqdm(range(total_sequences), desc="处理序列"):
        sequence = features_scaled[i]
        target_seq = target_scaled[i]

        for j in range(0, len(sequence) - window_size - 1, stride):
            X.append(sequence[j:j + window_size])
            y.append(target_seq[j + window_size])

    X = np.array(X).reshape(-1, window_size, 1)
    y = np.array(y).reshape(-1, 1)

    print(f"最终数据形状 - X: {X.shape}, y: {y.shape}")
    return X, y, scaler_features, scaler_target


# 优化的数据增强函数
def optimized_augment_data(X, y, noise_level=0.02):
    """更温和的数据增强"""
    X_aug, y_aug = [], []

    # 添加轻微高斯噪声
    noise = np.random.normal(0, noise_level, X.shape)
    X_noise = X + noise
    X_aug.append(X_noise)
    y_aug.append(y)

    # 时间扭曲（time warping）- 更适合时序数据
    X_warped = []
    for i in range(len(X)):
        seq = X[i, :, 0]
        # 创建一个轻微扭曲的索引
        indices = np.linspace(0, len(seq) - 1, len(seq))
        indices = indices + np.sin(np.linspace(0, 2 * np.pi, len(seq))) * 1.5
        indices = np.clip(indices, 0, len(seq) - 1).astype(int)
        warped_seq = seq[indices]
        X_warped.append(warped_seq.reshape(-1, 1))

    X_aug.append(np.array(X_warped).reshape(-1, X.shape[1], 1))
    y_aug.append(y)

    # 合并原始数据和增强数据
    X_combined = np.vstack([X] + X_aug)
    y_combined = np.vstack([y] + y_aug)

    # 随机打乱
    indices = np.random.permutation(len(X_combined))
    X_combined = X_combined[indices]
    y_combined = y_combined[indices]

    return X_combined, y_combined


# 优化的训练函数
def train_with_warmup_and_cosine_annealing(model, train_loader, val_loader, num_epochs, device, patience=25):
    """使用预热和余弦退火的训练策略"""
    criterion = BalancedLoss(mse_weight=0.6, mae_weight=0.3, peak_weight=0.1, peak_threshold=0.75)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.005)

    # 预热阶段 + 余弦退火
    warmup_epochs = int(0.1 * num_epochs)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=num_epochs - warmup_epochs,
        T_mult=1,
        eta_min=1e-6
    )

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    no_improve = 0

    epoch_pbar = trange(num_epochs, desc="训练中")

    for epoch in epoch_pbar:
        # 预热阶段手动调整学习率
        if epoch < warmup_epochs:
            lr = 0.0005 * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # 训练阶段
        model.train()
        train_loss = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [训练]", leave=False)
        for batch_x, batch_y in train_pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)

            loss = criterion(outputs, batch_y)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            train_pbar.set_postfix({'损失': f'{loss.item():.6f}'})

        # 验证阶段
        model.eval()
        val_loss = 0

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [验证]", leave=False)
        with torch.no_grad():
            for batch_x, batch_y in val_pbar:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                val_batch_loss = criterion(outputs, batch_y).item()
                val_loss += val_batch_loss

                val_pbar.set_postfix({'损失': f'{val_batch_loss:.6f}'})

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # 余弦退火阶段
        if epoch >= warmup_epochs:
            scheduler.step()

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
            torch.save(model.state_dict(), 'best_model.pth')
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
            outputs = model(batch_x)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(batch_y.numpy())

    # 转换为numpy数组
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # 创建一个新的RobustScaler，只用于单个特征
    new_scaler = RobustScaler()
    # 使用原始scaler的参数来设置新scaler
    new_scaler.center_ = np.array([scaler_target.center_[0]])
    new_scaler.scale_ = np.array([scaler_target.scale_[0]])
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
    plt.savefig('optimized_model_results.png', dpi=300, bbox_inches='tight')
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
    plt.savefig('optimized_model_last_samples.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    # 优化的参数设置
    WINDOW_SIZE = 64
    STRIDE = 8  # 稍微增加步长，减少样本数量但保持代表性
    BATCH_SIZE = 128
    NUM_EPOCHS = 150  # 减少轮数但使用更好的学习率调度

    print("开始数据预处理...")
    X, y, scaler_features, scaler_target = optimized_load_and_preprocess_data(
        'xtrain_new.csv',
        'ytrain_new.csv',
        WINDOW_SIZE,
        STRIDE
    )

    print("执行数据增强...")
    X_augmented, y_augmented = optimized_augment_data(X, y, noise_level=0.02)
    print(f"增强后的数据形状 - X: {X_augmented.shape}, y: {y_augmented.shape}")

    print("划分数据集...")
    # 使用分层抽样保证训练集和测试集分布一致
    X_train, X_test, y_train, y_test = train_test_split(
        X_augmented, y_augmented, test_size=0.15, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42
    )

    print("创建数据加载器...")
    train_dataset = SeismicDataset(X_train, y_train)
    val_dataset = SeismicDataset(X_val, y_val)
    test_dataset = SeismicDataset(X_test, y_test)

    # 根据系统设置worker数
    num_workers = 4  # 如果有多核CPU，可以设置为4或更高

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=num_workers)

    print("初始化优化的组合模型...")
    model = OptimizedCombinedModel(
        seq_len=WINDOW_SIZE,
        hidden_size=128,
        lstm_layers=2,  # 减少LSTM层数
        dropout=0.2  # 减少dropout
    ).to(device)

    # 打印模型结构与参数量
    print("\n模型架构:")
    print(model)
    print(f"\n总参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练模型
    print("\n开始训练...")
    train_losses, val_losses = train_with_warmup_and_cosine_annealing(
        model, train_loader, val_loader, NUM_EPOCHS, device, patience=25
    )

    print("加载最佳模型...")
    model.load_state_dict(torch.load('best_model.pth'))

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