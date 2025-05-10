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


# 多尺度特征提取器
class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, input_channels=1):
        super().__init__()
        self.conv_scales = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_channels, 32, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(32),
                nn.GELU(),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.GELU()
            ) for k in [3, 5, 9]  # 多尺度卷积核
        ])

        # 通道注意力机制
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(64 * 3, 32, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(32, 64 * 3, kernel_size=1),
            nn.Sigmoid()
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Conv1d(64 * 3, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.GELU()
        )

    def forward(self, x):
        # 应用多尺度卷积
        multi_features = []
        for conv in self.conv_scales:
            multi_features.append(conv(x))

        # 拼接多尺度特征
        features = torch.cat(multi_features, dim=1)  # [B, 64*3, L]

        # 应用通道注意力
        att = self.channel_attention(features)
        features = features * att

        # 特征融合
        return self.fusion(features)  # [B, 128, L]


# 增强的Transformer编码器
class EnhancedTransformerEncoder(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=4, dim_ff=512, dropout=0.2):
        super().__init__()

        # 使用PyTorch内置的TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True  # 先归一化再注意力，更稳定
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 全局上下文编码
        self.global_context = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # x: [B, L, D]
        # 应用Transformer
        out = self.transformer(x, mask=mask)

        # 全局上下文
        global_repr = torch.mean(out, dim=1)
        global_context = self.global_context(global_repr)

        return out, global_context


# 增强的地震模型
class EnhancedSeismicModel(nn.Module):
    def __init__(self, seq_len=36, nhead=8, num_layers=4, dropout=0.2):
        super().__init__()

        # 多尺度特征提取
        self.feature_extractor = MultiScaleFeatureExtractor(input_channels=1)

        # 位置编码
        self.register_buffer("position_ids", torch.arange(seq_len).expand(1, -1))
        self.position_embeddings = nn.Embedding(seq_len, 128)
        self.dropout = nn.Dropout(dropout)

        # Transformer编码器
        self.transformer = EnhancedTransformerEncoder(
            d_model=128,
            nhead=nhead,
            num_layers=num_layers,
            dim_ff=512,
            dropout=dropout
        )

        # 预测头
        self.prediction_head = nn.Sequential(
            nn.Linear(128 + 64, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: [B, L, 1]
        batch_size, seq_len, _ = x.shape

        # CNN特征提取 - 需要调整维度
        x_cnn = x.transpose(1, 2)  # [B, 1, L]
        features = self.feature_extractor(x_cnn)  # [B, 128, L]
        features = features.transpose(1, 2)  # [B, L, 128]

        # 添加位置编码
        position_embeddings = self.position_embeddings(self.position_ids)  # [1, L, 128]
        features = features + position_embeddings
        features = self.dropout(features)

        # Transformer编码
        transformer_out, global_context = self.transformer(features)

        # 获取最后时间步表示
        last_hidden = transformer_out[:, -1]

        # 合并全局上下文和最后隐藏状态
        combined = torch.cat([last_hidden, global_context], dim=1)

        # 预测
        return self.prediction_head(combined)


def load_and_preprocess_data(features_path, target_path, window_size=36, stride=12):
    """增强的数据预处理函数"""
    print("加载数据...")
    features = pd.read_csv(features_path, header=None).T
    target = pd.read_csv(target_path, header=None).T

    print(f"原始数据形状 - 特征: {features.shape}, 目标: {target.shape}")

    print("预处理数据...")
    # 标准化比归一化更适合时间序列数据
    scaler_features = StandardScaler()
    scaler_target = StandardScaler()

    features_scaled = scaler_features.fit_transform(features)
    target_scaled = scaler_target.fit_transform(target)

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


def train_model(model, train_loader, val_loader, num_epochs, device, patience=15):
    """增强的训练函数，包含早停和余弦退火"""
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.02)

    # 余弦退火学习率
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    no_improve = 0

    epoch_pbar = trange(num_epochs, desc="训练中")

    for epoch in epoch_pbar:
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

            # 梯度裁剪，防止梯度爆炸
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

        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

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

    # 创建一个新的StandardScaler，只用于单个特征
    new_scaler = StandardScaler()
    # 使用原始scaler的参数来设置新scaler
    new_scaler.mean_ = np.array([scaler_target.mean_[0]])  # 取第一个特征的参数
    new_scaler.scale_ = np.array([scaler_target.scale_[0]])  # 取第一个特征的参数
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
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
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
    plt.savefig('last_500_samples.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    # 参数设置 - 优化后
    WINDOW_SIZE = 36  # 增加窗口大小
    STRIDE = 12  # 添加窗口滑动步长
    BATCH_SIZE = 128  # 批次大小，根据GPU内存调整
    NUM_EPOCHS = 150  # 增加最大训练轮数

    print("开始数据预处理...")
    # 使用优化后的数据预处理函数
    X, y, scaler_features, scaler_target = load_and_preprocess_data(
        'xtrain_new.csv',
        'ytrain_new.csv',
        WINDOW_SIZE,
        STRIDE
    )

    print("划分数据集...")
    # 划分数据集 - 保持时间序列顺序
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

    # 根据系统可用核心数调整num_workers
    num_workers = 0  # 如果有多核CPU，可以设置为4或更高

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=num_workers)

    print("初始化增强模型...")
    # 使用增强的模型
    model = EnhancedSeismicModel(
        seq_len=WINDOW_SIZE,
        nhead=8,
        num_layers=4,
        dropout=0.2
    ).to(device)

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
    # 加载最佳模型
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

# 36轮后触发早停！
# 加载最佳模型...
# 评估模型...
# 评估中: 100%|██████████| 209/209 [00:02<00:00, 87.83it/s]
#
# === 模型评估结果 ===
# R2 分数: 0.7864
# RMSE: 0.0000
# MAE: 0.0000
