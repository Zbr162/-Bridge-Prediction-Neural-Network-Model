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
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, Batch

warnings.filterwarnings("ignore")

# 设置随机种子
torch.manual_seed(68)
np.random.seed(68)

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['STSong']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

# 检查CUDA是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


# 自定义图数据集类
class SeismicGraphDataset(Dataset):
    def __init__(self, features, targets, window_size=64, k_neighbors=8, time_decay=0.1):
        """
        Args:
            features: 特征数据 [samples, seq_len]
            targets: 目标数据 [samples]
            window_size: 窗口大小，即每个图的节点数
            k_neighbors: 构建图时每个节点连接的邻居数
            time_decay: 时间衰减系数，控制时间距离对边权重的影响
        """
        self.features = features
        self.targets = targets.reshape(-1, 1)
        self.window_size = window_size
        self.k_neighbors = k_neighbors
        self.time_decay = time_decay
        self.num_samples = features.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 获取样本特征和目标
        x = self.features[idx].reshape(self.window_size, 1)  # [window_size, 1]
        y = self.targets[idx]  # [1]

        # 创建节点特征矩阵
        node_features = []
        for i in range(self.window_size):
            # 基础特征: 当前值
            node_feat = [x[i, 0]]

            # 添加位置编码
            pos_encoding = i / self.window_size
            node_feat.append(pos_encoding)

            # 可以添加更多特征，如局部统计量
            if i >= 2:
                # 局部趋势
                local_diff = x[i, 0] - x[i - 1, 0]
                local_diff2 = x[i - 1, 0] - x[i - 2, 0]
                node_feat.extend([local_diff, local_diff2])
            else:
                node_feat.extend([0, 0])

            node_features.append(node_feat)

        # 转换为PyTorch张量
        node_features = torch.FloatTensor(node_features)  # [window_size, node_feature_dim]

        # 构建图的邻接关系
        edge_index, edge_attr = self._build_graph(x)

        # 创建PyG数据对象
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.FloatTensor([y])
        )

        return data

    def _build_graph(self, x):
        """构建图结构：结合时序关系和值相似性"""
        node_values = x.flatten()
        num_nodes = len(node_values)

        # 1. 时序关系连接 (连接相邻时间点)
        temporal_edges = []
        for i in range(num_nodes - 1):
            temporal_edges.append((i, i + 1))
            temporal_edges.append((i + 1, i))  # 双向连接

            # 也可以连接更远的时间点 (如跳跃连接)
            if i < num_nodes - 2:
                temporal_edges.append((i, i + 2))
                temporal_edges.append((i + 2, i))

        # 2. 基于值相似性的连接
        similarity_edges = []
        similarity_weights = []

        # 计算所有节点对之间的相似性
        for i in range(num_nodes):
            # 计算与其他节点的值差异
            diffs = np.abs(node_values[i] - node_values)

            # 结合时间距离因素
            time_dists = np.abs(np.arange(num_nodes) - i)
            time_weights = np.exp(-self.time_decay * time_dists)

            # 综合得分 (值差异小且时间距离不太远的点得分高)
            combined_scores = -diffs * time_weights

            # 选择top-k个邻居 (除自己外)
            combined_scores[i] = float('-inf')  # 排除自身
            topk_indices = np.argsort(combined_scores)[-self.k_neighbors:]

            for j in topk_indices:
                if i != j:  # 确保不自连接
                    similarity_edges.append((i, j))
                    # 使用相似度作为边权重
                    weight = 1.0 / (1.0 + np.abs(node_values[i] - node_values[j]))
                    similarity_weights.append(weight)

        # 合并两种边
        all_edges = temporal_edges + similarity_edges

        # 边权重：时序边权重为1，相似性边权重为计算值
        temporal_weights = [1.0] * len(temporal_edges)
        all_weights = temporal_weights + similarity_weights

        # 转换为PyTorch张量
        edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(all_weights, dtype=torch.float).view(-1, 1)

        return edge_index, edge_attr


# 定义时间嵌入层
class TimeEncoder(nn.Module):
    def __init__(self, window_size, embedding_dim):
        super(TimeEncoder, self).__init__()
        self.linear = nn.Linear(1, embedding_dim)

        # 创建固定的位置编码
        time_steps = torch.arange(0, window_size, 1).float() / window_size
        time_steps = time_steps.view(-1, 1)
        self.register_buffer('time_steps', time_steps)

    def forward(self, batch_size):
        # 为每个批次创建时间编码
        # batch_size x window_size x embedding_dim
        time_encoding = self.linear(self.time_steps)
        time_encoding = time_encoding.unsqueeze(0).repeat(batch_size, 1, 1)
        return time_encoding


# 定义图注意力层
class GraphAttentionBlock(nn.Module):
    def __init__(self, in_dim, out_dim, heads=8, dropout=0.1):
        super(GraphAttentionBlock, self).__init__()
        self.gat = GATConv(in_dim, out_dim, heads=heads, dropout=dropout)
        self.norm = nn.LayerNorm(out_dim * heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr=None):
        # 应用GAT
        x_gat = self.gat(x, edge_index)
        # 应用LayerNorm和Dropout
        x_gat = self.norm(x_gat)
        x_gat = self.dropout(x_gat)
        return x_gat


# 定义图卷积块
class GraphConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, aggr='mean'):
        super(GraphConvBlock, self).__init__()
        self.gcn = GCNConv(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.activation = nn.GELU()

    def forward(self, x, edge_index, edge_attr=None):
        # 应用GCN
        x_gcn = self.gcn(x, edge_index, edge_weight=edge_attr.squeeze() if edge_attr is not None else None)
        # 应用LayerNorm和激活函数
        x_gcn = self.norm(x_gcn)
        x_gcn = self.activation(x_gcn)
        return x_gcn


# 定义残差图卷积块
class ResGraphBlock(nn.Module):
    def __init__(self, hidden_dim):
        super(ResGraphBlock, self).__init__()
        self.gcn1 = GraphConvBlock(hidden_dim, hidden_dim)
        self.gcn2 = GraphConvBlock(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_attr=None):
        residual = x
        out = self.gcn1(x, edge_index, edge_attr)
        out = self.gcn2(out, edge_index, edge_attr)
        return out + residual


# 定义频域特征提取模块
class FrequencyFeatureExtractor(nn.Module):
    def __init__(self, window_size, hidden_dim):
        super(FrequencyFeatureExtractor, self).__init__()
        self.freq_dim = window_size // 2 + 1
        self.projection = nn.Sequential(
            nn.Linear(self.freq_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def forward(self, x_batch):
        # 输入x_batch是批次的时序数据: [batch_size, window_size]
        # 计算FFT
        x_fft = torch.abs(torch.fft.rfft(x_batch, dim=1))  # [batch_size, window_size//2 + 1]

        # 投影到隐藏维度
        freq_features = self.projection(x_fft)  # [batch_size, hidden_dim]
        return freq_features


# 定义时空图神经网络模型
class STGraphNet(nn.Module):
    def __init__(
            self,
            node_feature_dim=4,  # 初始节点特征维度
            hidden_dim=256,  # 隐藏层维度
            window_size=64,  # 窗口大小
            num_gnn_layers=4,  # GNN层数
            dropout=0.2,  # Dropout比例
            heads=8  # 多头注意力头数
    ):
        super(STGraphNet, self).__init__()

        # 节点特征嵌入
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 图注意力层
        self.gat_layer = GraphAttentionBlock(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout)

        # 多层图卷积
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_gnn_layers):
            self.gnn_layers.append(ResGraphBlock(hidden_dim))

        # 图读出层（用于聚合整个图的信息）
        self.graph_readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 频域特征提取器
        self.freq_extractor = FrequencyFeatureExtractor(window_size, hidden_dim)

        # 输出预测头
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        # 保存窗口大小，用于频域特征提取
        self.window_size = window_size

    def forward(self, data):
        # 对于批处理的图数据
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # 1. 编码节点特征
        x = self.node_encoder(x)

        # 2. 应用图注意力层
        x = self.gat_layer(x, edge_index, edge_attr)

        # 3. 应用多层图卷积
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, edge_index, edge_attr)

        # 4. 图池化：聚合所有节点信息
        # 使用mean pooling汇总每个图的节点特征
        graph_embedding = global_mean_pool(x, batch)
        graph_features = self.graph_readout(graph_embedding)

        # 5. 提取频域特征
        # 收集每个图的原始时序数据用于FFT
        batch_size = torch.max(batch).item() + 1
        time_series = torch.zeros(batch_size, self.window_size).to(x.device)

        for i in range(batch_size):
            # 获取当前图的节点索引
            node_indices = (batch == i).nonzero().view(-1)
            # 获取节点原始值（假设第一列是值）
            node_values = data.x[node_indices, 0]
            time_series[i] = node_values

        # 提取频域特征
        freq_features = self.freq_extractor(time_series)

        # 6. 融合特征并预测
        combined_features = torch.cat([graph_features, freq_features], dim=1)
        output = self.prediction_head(combined_features)

        return output


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


# 定义用于将批次数据转换为PyG批次的函数
def collate_fn(batch):
    return Batch.from_data_list(batch)


# 优化的数据预处理函数，适用于图模型
def load_and_preprocess_data_for_graph(features_path, target_path, window_size=64, stride=8):
    """为图模型预处理数据"""
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

    # 使用滑动窗口生成样本
    for i in tqdm(range(total_sequences), desc="处理序列"):
        sequence = features_scaled[i]
        target_seq = target_scaled[i]

        for j in range(0, len(sequence) - window_size - 1, stride):
            X.append(sequence[j:j + window_size])
            y.append(target_seq[j + window_size])

    X = np.array(X)
    y = np.array(y)

    print(f"最终数据形状 - X: {X.shape}, y: {y.shape}")
    return X, y, scaler_features, scaler_target


def augment_data_for_graph(X, y, noise_level=0.02):
    """数据增强，适用于图模型"""
    # 确保y是一维数组
    y = y.reshape(-1)

    X_aug, y_aug = [], []

    # 添加轻微高斯噪声
    noise = np.random.normal(0, noise_level, X.shape)
    X_noise = X + noise
    X_aug.append(X_noise)
    y_aug.append(y.reshape(-1, 1))  # 确保是二维数组

    # 时间扭曲（time warping）- 更适合时序数据
    X_warped = []
    for i in range(len(X)):
        seq = X[i]
        # 创建一个轻微扭曲的索引
        indices = np.linspace(0, len(seq) - 1, len(seq))
        indices = indices + np.sin(np.linspace(0, 2 * np.pi, len(seq))) * 1.5
        indices = np.clip(indices, 0, len(seq) - 1).astype(int)
        warped_seq = seq[indices]
        X_warped.append(warped_seq)

    X_aug.append(np.array(X_warped))
    y_aug.append(y.reshape(-1, 1))  # 确保是二维数组

    # 合并原始数据和增强数据
    X_combined = np.vstack([X] + X_aug)
    y_combined = np.vstack([y.reshape(-1, 1)] + y_aug).reshape(-1)

    # 随机打乱
    indices = np.random.permutation(len(X_combined))
    X_combined = X_combined[indices]
    y_combined = y_combined[indices]

    return X_combined, y_combined


# 训练函数
def train_graph_model(model, train_loader, val_loader, num_epochs, device, patience=25):
    """训练图模型"""
    criterion = BalancedLoss(mse_weight=0.6, mae_weight=0.3, peak_weight=0.1, peak_threshold=0.75)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.005)

    # 学习率调度器：预热+余弦退火
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
        for batch in train_pbar:
            batch = batch.to(device)
            optimizer.zero_grad()

            outputs = model(batch)
            loss = criterion(outputs, batch.y)
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
            for batch in val_pbar:
                batch = batch.to(device)
                outputs = model(batch)
                val_batch_loss = criterion(outputs, batch.y).item()
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
            torch.save(model.state_dict(), 'best_graph_model.pth')
            no_improve = 0
        else:
            no_improve += 1

        # 早停
        if no_improve >= patience:
            print(f"\n{epoch + 1}轮后触发早停！")
            break

    return train_losses, val_losses


# 评估函数
def evaluate_graph_model(model, test_loader, scaler_target, device):
    """评估图模型"""
    model.eval()
    predictions = []
    actuals = []

    # 使用tqdm显示评估进度
    test_pbar = tqdm(test_loader, desc="评估中")
    with torch.no_grad():
        for batch in test_pbar:
            batch = batch.to(device)
            outputs = model(batch)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(batch.y.cpu().numpy())

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
def plot_graph_results(results, train_losses, val_losses):
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
    plt.savefig('graph_model_results.png', dpi=300, bbox_inches='tight')
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
    plt.savefig('graph_model_last_samples.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    # 设置参数
    WINDOW_SIZE = 64
    STRIDE = 6
    BATCH_SIZE = 192  # 对于GNN，可能需要减小批次大小
    NUM_EPOCHS = 150

    print("开始数据预处理...")
    X, y, scaler_features, scaler_target = load_and_preprocess_data_for_graph(
        'xtrain_new.csv',
        'ytrain_new.csv',
        WINDOW_SIZE,
        STRIDE
    )

    print("执行数据增强...")
    X_augmented, y_augmented = augment_data_for_graph(X, y, noise_level=0.02)
    print(f"增强后的数据形状 - X: {X_augmented.shape}, y: {y_augmented.shape}")

#    X_augmented, y_augmented = X, y  # 跳过数据增强

    print("划分数据集...")
    # 使用分层抽样保证训练集和测试集分布一致
    X_train, X_test, y_train, y_test = train_test_split(
        X_augmented, y_augmented, test_size=0.15, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42
    )

    print("创建图数据集...")
    train_dataset = SeismicGraphDataset(X_train, y_train, window_size=WINDOW_SIZE)
    val_dataset = SeismicGraphDataset(X_val, y_val, window_size=WINDOW_SIZE)
    test_dataset = SeismicGraphDataset(X_test, y_test, window_size=WINDOW_SIZE)

    # 根据系统设置worker数
    num_workers = 4  # 如果有多核CPU，可以设置为4或更高

    print("创建数据加载器...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    print("初始化图神经网络模型...")
    model = STGraphNet(
        node_feature_dim=4,  # 节点特征维度: 值+位置编码+2个局部差分特征
        hidden_dim=256,
        window_size=WINDOW_SIZE,
        num_gnn_layers=4,
        dropout=0.2,
        heads=8
    ).to(device)

    # 打印模型结构与参数量
    print("\n模型架构:")
    print(model)
    print(f"\n总参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练模型
    print("\n开始训练...")
    train_losses, val_losses = train_graph_model(
        model, train_loader, val_loader, NUM_EPOCHS, device, patience=25
    )

    print("加载最佳模型...")
    model.load_state_dict(torch.load('best_graph_model.pth'))

    # 评估模型
    print("评估模型...")
    results = evaluate_graph_model(model, test_loader, scaler_target, device)

    print("\n=== 图神经网络模型评估结果 ===")
    print(f"R2 分数: {results['R2']:.4f}")
    print(f"RMSE: {results['RMSE']:.4f}")
    print(f"MAE: {results['MAE']:.4f}")

    # 绘制结果
    print("\n绘制结果...")
    plot_graph_results(results, train_losses, val_losses)

    print("\n训练完成!")


if __name__ == '__main__':
    main()
# GNN_V1 ++