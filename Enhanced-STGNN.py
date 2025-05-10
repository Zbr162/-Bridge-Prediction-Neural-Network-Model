import torch
# 引入数学库
import math
# 导入GAT注意力层的相关工具
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import softmax
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import os
import warnings
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.data import Data, Batch
from torch_scatter import scatter_mean, scatter_max, scatter_add
from scipy.signal import find_peaks


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


# 自定义图数据集类 - 增强版
class SeismicGraphDataset(Dataset):
    def __init__(self, features, targets, window_size=64, k_neighbors=12, time_decay=0.05, include_spectral=True):
        """
        Args:
            features: 特征数据 [samples, seq_len]
            targets: 目标数据 [samples]
            window_size: 窗口大小，即每个图的节点数
            k_neighbors: 构建图时每个节点连接的邻居数
            time_decay: 时间衰减系数，控制时间距离对边权重的影响
            include_spectral: 是否包含谱特征
        """
        self.features = features
        self.targets = targets.reshape(-1, 1)
        self.window_size = window_size
        self.k_neighbors = k_neighbors
        self.time_decay = time_decay
        self.num_samples = features.shape[0]
        self.include_spectral = include_spectral

        # 预计算所有样本的FFT特征，以加速训练
        if include_spectral:
            self.fft_features = []
            for i in range(self.num_samples):
                fft = np.abs(np.fft.rfft(self.features[i]))
                self.fft_features.append(fft)
            self.fft_features = np.array(self.fft_features)

            # 规范化FFT特征
            self.fft_scaler = MinMaxScaler()
            self.fft_features = self.fft_scaler.fit_transform(self.fft_features)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 获取样本特征和目标
        x = self.features[idx].reshape(self.window_size, 1)  # [window_size, 1]
        y = self.targets[idx]  # [1]

        # 创建增强的节点特征矩阵
        node_features = []
        for i in range(self.window_size):
            # 基础特征: 当前值
            node_feat = [x[i, 0]]

            # 添加位置编码
            pos_encoding = i / self.window_size
            node_feat.append(pos_encoding)

            # 添加周期性位置编码
            sin_pos = np.sin(2 * np.pi * pos_encoding)
            cos_pos = np.cos(2 * np.pi * pos_encoding)
            node_feat.extend([sin_pos, cos_pos])

            # 添加局部统计特征
            if i >= 2:
                # 局部趋势
                local_diff = x[i, 0] - x[i - 1, 0]
                local_diff2 = x[i - 1, 0] - x[i - 2, 0]
                # 局部加速度（二阶差分）
                local_accel = local_diff - local_diff2
                node_feat.extend([local_diff, local_diff2, local_accel])
            else:
                node_feat.extend([0, 0, 0])

            # 添加局部波动特征
            if i >= 5:
                # 局部5点窗口统计
                local_window = x[max(0, i - 5):i + 1, 0]
                local_mean = np.mean(local_window)
                local_std = np.std(local_window)
                local_max = np.max(local_window)
                local_min = np.min(local_window)
                local_range = local_max - local_min
                node_feat.extend([local_mean, local_std, local_range])
            else:
                node_feat.extend([x[i, 0], 0, 0])  # 对于前5个点用当前值填充

            node_features.append(node_feat)

        # 转换为PyTorch张量
        node_features = torch.FloatTensor(node_features)  # [window_size, node_feature_dim]

        # 构建图的邻接关系
        edge_index, edge_attr = self._build_graph(x)

        # 添加全局特征
        global_features = self._extract_global_features(x)

        # 添加谱特征
        if self.include_spectral:
            spectral_features = torch.FloatTensor(self.fft_features[idx])
        else:
            spectral_features = None

        # 创建PyG数据对象
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.FloatTensor([y]),
            global_features=torch.FloatTensor(global_features),
            spectral_features=spectral_features if spectral_features is not None else torch.zeros(1)
        )

        return data

    def _extract_global_features(self, x):
        """提取序列的全局特征"""
        x_flat = x.flatten()

        # 统计特征
        mean = np.mean(x_flat)
        std = np.std(x_flat)
        max_val = np.max(x_flat)
        min_val = np.min(x_flat)

        # 趋势特征
        slope = np.polyfit(np.arange(len(x_flat)), x_flat, 1)[0]

        # 波动特征
        zero_crossings = np.sum(np.diff(np.signbit(x_flat - mean)))

        # 峰值特征
        peaks, _ = find_peaks(x_flat, height=mean)
        peak_count = len(peaks)

        return [mean, std, max_val, min_val, max_val - min_val, slope, zero_crossings / len(x_flat),
                peak_count / len(x_flat)]

    def _build_graph(self, x):
        """构建增强型图结构：结合时序关系、值相似性和多尺度连接"""
        node_values = x.flatten()
        num_nodes = len(node_values)
        all_edges = []
        all_weights = []

        # 1. 时序关系连接 (多尺度连接)
        for scale in [1, 2, 3]:  # 多尺度：直接相邻、隔1个连接、隔2个连接
            for i in range(num_nodes - scale):
                all_edges.append((i, i + scale))
                all_edges.append((i + scale, i))  # 双向连接
                all_weights.extend([1.0 / scale, 1.0 / scale])  # 根据跨度调整权重

        # 2. 基于值相似性的连接
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
                    all_edges.append((i, j))
                    # 使用相似度作为边权重
                    weight = 1.0 / (1.0 + np.abs(node_values[i] - node_values[j]))
                    all_weights.append(weight)

        # 3. 长距离连接（增强全局信息流动）
        segment_size = num_nodes // 4  # 将序列分为4个部分
        for seg_idx in range(4):
            start_idx = seg_idx * segment_size
            end_idx = (seg_idx + 1) * segment_size if seg_idx < 3 else num_nodes

            # 计算每段的代表点（均值最接近的点）
            segment = node_values[start_idx:end_idx]
            segment_mean = np.mean(segment)
            representative_idx = start_idx + np.argmin(np.abs(segment - segment_mean))

            # 连接不同段的代表点
            for other_seg in range(4):
                if other_seg != seg_idx:
                    other_start = other_seg * segment_size
                    other_end = (other_seg + 1) * segment_size if other_seg < 3 else num_nodes
                    other_segment = node_values[other_start:other_end]
                    other_mean = np.mean(other_segment)
                    other_rep_idx = other_start + np.argmin(np.abs(other_segment - other_mean))

                    all_edges.append((representative_idx, other_rep_idx))
                    all_edges.append((other_rep_idx, representative_idx))
                    all_weights.extend([0.5, 0.5])  # 长距离连接权重较低

        # 转换为PyTorch张量
        edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(all_weights, dtype=torch.float).view(-1, 1)

        return edge_index, edge_attr


# 自注意力机制
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        # 确保hidden_dim能被num_heads整除
        assert hidden_dim % num_heads == 0
        self.head_dim = hidden_dim // num_heads

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def transpose_for_scores(self, x):
        # x: [batch_size, seq_len, hidden_dim]
        batch_size, seq_len = x.size(0), x.size(1)
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        # 转置为 [batch_size, num_heads, seq_len, head_dim]
        return x.permute(0, 2, 1, 3)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        # 残差连接
        residual = x

        # 线性投影
        q = self.query(x)  # [batch_size, seq_len, hidden_dim]
        k = self.key(x)  # [batch_size, seq_len, hidden_dim]
        v = self.value(x)  # [batch_size, seq_len, hidden_dim]

        # 重塑为多头形式
        q = self.transpose_for_scores(q)  # [batch_size, num_heads, seq_len, head_dim]
        k = self.transpose_for_scores(k)  # [batch_size, num_heads, seq_len, head_dim]
        v = self.transpose_for_scores(v)  # [batch_size, num_heads, seq_len, head_dim]

        # 注意力分数
        # [batch_size, num_heads, seq_len, seq_len]
        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / (self.head_dim ** 0.5)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # 注意力权重
        attention_weights = torch.softmax(attention_scores, dim=-1)  # [batch_size, num_heads, seq_len, seq_len]
        attention_weights = self.dropout(attention_weights)

        # 注意力输出
        context = torch.matmul(attention_weights, v)  # [batch_size, num_heads, seq_len, head_dim]
        # 转换回原始形状
        context = context.permute(0, 2, 1, 3).contiguous()  # [batch_size, seq_len, num_heads, head_dim]
        context = context.view(batch_size, seq_len, -1)  # [batch_size, seq_len, hidden_dim]

        # 输出投影
        output = self.out_proj(context)  # [batch_size, seq_len, hidden_dim]
        output = self.dropout(output)

        # 残差连接和层归一化
        output = self.layer_norm(output + residual)  # [batch_size, seq_len, hidden_dim]

        return output


# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len=1000):
        super(PositionalEncoding, self).__init__()

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))

        # 正弦和余弦位置编码
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, hidden_dim]

        # 注册为buffer (不作为模型参数)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, hidden_dim]
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


# Transformer编码器层
class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, ff_dim=512, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = SelfAttention(hidden_dim, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 自注意力子层
        attn_output = self.self_attn(x, mask)

        # 前馈网络子层
        ff_output = self.feed_forward(self.norm1(attn_output))
        output = self.norm2(attn_output + ff_output)

        return output


# 多尺度图卷积模块
class MultiScaleGraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, scales=[1, 2, 4]):
        super(MultiScaleGraphConv, self).__init__()
        self.scales = scales
        self.convs = nn.ModuleList()

        # 每个尺度一个GCN
        for _ in scales:
            self.convs.append(GCNConv(in_dim, out_dim // len(scales)))

        # 输出投影层
        self.projection = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU()
        )

    def forward(self, x, edge_index, edge_attr=None):
        # 收集每个尺度的结果
        multi_scale_features = []

        for i, scale in enumerate(self.scales):
            if scale == 1:
                # 标准图卷积
                scale_feat = self.convs[i](x, edge_index,
                                           edge_weight=edge_attr.squeeze() if edge_attr is not None else None)
            else:
                # 计算k次幂图
                # 这里简化处理，实际上应该计算邻接矩阵的幂
                # 但为了效率，我们可以简单地多次应用GCN
                scale_feat = x
                for _ in range(scale):
                    scale_feat = self.convs[i](scale_feat, edge_index,
                                               edge_weight=edge_attr.squeeze() if edge_attr is not None else None)

            multi_scale_features.append(scale_feat)

        # 拼接所有尺度的特征
        output = torch.cat(multi_scale_features, dim=1)
        # 投影回原始维度
        output = self.projection(output)

        return output


# 改进的图注意力层，增加多头注意力
class ImprovedGraphAttention(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.1, edge_dim=1):
        super(ImprovedGraphAttention, self).__init__()
        self.gat = GATConv(in_dim, out_dim // heads, heads=heads, dropout=dropout, edge_dim=edge_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

        # 节点级注意力
        self.node_attention = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.GELU(),
            nn.Linear(out_dim // 2, 1),
        )

    def forward(self, x, edge_index, edge_attr=None):
        # 应用GAT
        x_gat = self.gat(x, edge_index, edge_attr)

        # 计算节点级注意力权重
        attn_weights = torch.sigmoid(self.node_attention(x_gat))  # [num_nodes, 1]

        # 应用注意力
        x_weighted = x_gat * attn_weights

        # 归一化和Dropout
        x_out = self.norm(x_weighted)
        x_out = self.dropout(x_out)

        return x_out, attn_weights


# 改进的残差图卷积块
class ImprovedResGraphBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super(ImprovedResGraphBlock, self).__init__()
        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr=None):
        residual = x

        # 第一个图卷积层
        out = self.gcn1(x, edge_index, edge_weight=edge_attr.squeeze() if edge_attr is not None else None)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)

        # 第二个图卷积层
        out = self.gcn2(out, edge_index, edge_weight=edge_attr.squeeze() if edge_attr is not None else None)
        out = self.norm2(out)
        out = out + residual  # 残差连接
        out = self.activation(out)

        return out


# 改进的频域特征提取模块，使用注意力机制
class ImprovedFrequencyFeatureExtractor(nn.Module):
    def __init__(self, freq_dim, hidden_dim, num_heads=4):
        super(ImprovedFrequencyFeatureExtractor, self).__init__()
        self.freq_dim = freq_dim

        # 频率嵌入
        self.freq_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # 频域注意力
        self.freq_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=0.1)

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def forward(self, x_fft):
        # x_fft: [batch_size, freq_dim]

        # 频率嵌入
        freq_features = self.freq_embedding(x_fft)  # [batch_size, hidden_dim]

        # 为自注意力准备输入 (需要seq_len维度)
        freq_features = freq_features.unsqueeze(1)  # [batch_size, 1, hidden_dim]

        # 自注意力 (在批次内应用)
        # 需要将batch_first=False的格式转换为[seq_len, batch_size, hidden_dim]
        freq_features_t = freq_features.transpose(0, 1)  # [1, batch_size, hidden_dim]
        attn_output, _ = self.freq_attention(
            freq_features_t, freq_features_t, freq_features_t
        )
        attn_output = attn_output.transpose(0, 1)  # [batch_size, 1, hidden_dim]

        # 压缩seq_len维度
        attn_output = attn_output.squeeze(1)  # [batch_size, hidden_dim]

        # 输出层
        output = self.output_layer(attn_output)  # [batch_size, hidden_dim]

        return output


# 多种图池化组合
class MultiPooling(nn.Module):
    def __init__(self, hidden_dim):
        super(MultiPooling, self).__init__()
        self.projection = nn.Linear(hidden_dim * 3, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.GELU()

    def forward(self, x, batch):
        # 应用多种池化方法
        mean_pooled = global_mean_pool(x, batch)  # [batch_size, hidden_dim]
        max_pooled = global_max_pool(x, batch)  # [batch_size, hidden_dim]
        sum_pooled = global_add_pool(x, batch)  # [batch_size, hidden_dim]

        # 拼接多种池化结果
        pooled = torch.cat([mean_pooled, max_pooled, sum_pooled], dim=1)  # [batch_size, hidden_dim*3]

        # 投影回原始维度
        pooled = self.projection(pooled)  # [batch_size, hidden_dim]
        pooled = self.norm(pooled)
        pooled = self.activation(pooled)

        return pooled

# 最大限度保留峰值的卷积注意力层
class PeakPreservingAttention(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1, negative_slope=0.2):
        super(PeakPreservingAttention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 线性变换
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.att = nn.Parameter(torch.Tensor(1, out_channels * 2))
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        # 峰值检测层
        self.peak_detector = nn.Sequential(
            nn.Linear(in_channels, out_channels // 2),
            nn.GELU(),
            nn.Linear(out_channels // 2, 1),
            nn.Sigmoid()
        )

        # 激活和Dropout
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin.weight)
        glorot(self.att)
        zeros(self.bias)

        # 继续实现PeakPreservingAttention类的forward方法
    def forward(self, x, edge_index):
        # 线性变换获取节点特征
        x = self.lin(x)

        # 计算峰值权重
        peak_weights = self.peak_detector(x)

        # 准备源节点和目标节点特征
        row, col = edge_index
        src, dst = x[row], x[col]

        # 拼接源目标特征用于注意力计算
        alpha = torch.cat([src, dst], dim=1)
        # 应用注意力向量
        alpha = (alpha * self.att).sum(dim=-1)
        # LeakyReLU激活
        alpha = nn.functional.leaky_relu(alpha, self.negative_slope)
        # Softmax归一化
        alpha = softmax(alpha, row)
        # Dropout
        alpha = nn.functional.dropout(alpha, p=self.dropout, training=self.training)

        # 加权聚合
        out = torch.zeros_like(x)
        out.index_add_(0, row, src * alpha.view(-1, 1))

        # 加入峰值权重，增强峰值区域的表示
        out = out * (1 + peak_weights)

        # 添加偏置
        out = out + self.bias

        return out

# 定义增强型时空图神经网络模型
class EnhancedSTGraphNet(nn.Module):
    def __init__(
            self,
            node_feature_dim=12,  # 增强后的节点特征维度
            hidden_dim=192,  # 增加隐藏层维度
            freq_dim=33,  # 频域特征维度
            window_size=64,  # 窗口大小
            num_gnn_layers=4,  # 增加GNN层数
            dropout=0.2,  # Dropout比例
            heads=8  # 增加多头注意力头数
    ):
        super(EnhancedSTGraphNet, self).__init__()

        # 节点特征嵌入
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 全局特征处理
        self.global_encoder = nn.Sequential(
            nn.Linear(8, hidden_dim // 2),  # 8个全局特征
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 多尺度图卷积层
        self.multi_scale_conv = MultiScaleGraphConv(hidden_dim, hidden_dim)

        # 图注意力层
        self.gat_layer = ImprovedGraphAttention(hidden_dim, hidden_dim, heads=heads, dropout=dropout)

        # 峰值保留注意力层
        self.peak_attn = PeakPreservingAttention(hidden_dim, hidden_dim, dropout=dropout)

        # 多层图卷积
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_gnn_layers):
            self.gnn_layers.append(ImprovedResGraphBlock(hidden_dim, dropout))

        # 多种池化组合
        self.multi_pooling = MultiPooling(hidden_dim)

        # 改进的频域特征提取器
        self.freq_extractor = ImprovedFrequencyFeatureExtractor(freq_dim, hidden_dim, num_heads=4)

        # Transformer全局依赖建模
        self.transformer_encoder = TransformerEncoderLayer(hidden_dim, num_heads=heads, dropout=dropout)

        # 特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 输出预测头
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 保存窗口大小，用于频域特征提取
        self.window_size = window_size
        self.hidden_dim = hidden_dim

    def forward(self, data):
        # 对于批处理的图数据
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        global_features = data.global_features
        spectral_features = data.spectral_features

        # 1. 编码节点特征
        node_features = self.node_encoder(x)

        # 2. 应用多尺度图卷积
        multi_scale_features = self.multi_scale_conv(node_features, edge_index, edge_attr)

        # 3. 应用改进的图注意力层
        gat_features, node_attn_weights = self.gat_layer(multi_scale_features, edge_index, edge_attr)

        # 4. 应用峰值保留注意力
        peak_features = self.peak_attn(gat_features, edge_index)

        # 5. 融合多种特征
        x = gat_features + 0.3 * peak_features  # 加权融合

        # 6. 应用多层图卷积
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, edge_index, edge_attr)

        # 7. 多种池化聚合所有节点信息
        graph_embedding = self.multi_pooling(x, batch)  # [batch_size, hidden_dim]

        # 8. 编码全局特征
        batch_size = torch.max(batch).item() + 1 if batch.numel() > 0 else 0
        global_feat_batch = global_features.view(batch_size, -1)  # [batch_size, 8]
        encoded_global = self.global_encoder(global_feat_batch)  # [batch_size, hidden_dim//2]

        # 9. 处理频域特征
        freq_features = self.freq_extractor(spectral_features)  # [batch_size, hidden_dim]

        # 10. 转换图嵌入为序列，应用Transformer
        # 为了应用Transformer，我们需要将图嵌入重塑为序列形式
        # 这里我们通过复制隐藏状态来创建一个"假"序列
        seq_len = min(16, self.window_size // 4)  # 使用更短的序列长度提高效率
        graph_seq = graph_embedding.unsqueeze(1).repeat(1, seq_len, 1)  # [batch_size, seq_len, hidden_dim]

        # 应用Transformer编码器
        trans_output = self.transformer_encoder(graph_seq)  # [batch_size, seq_len, hidden_dim]
        # 取序列的均值作为全局表示
        trans_embedding = torch.mean(trans_output, dim=1)  # [batch_size, hidden_dim]

        # 11. 融合三种特征：图特征、全局特征和频域特征
        # 扩展全局特征维度以便于拼接
        encoded_global_expanded = torch.zeros(batch_size, self.hidden_dim, device=encoded_global.device)
        encoded_global_expanded[:, :self.hidden_dim // 2] = encoded_global

        combined_features = torch.cat([
            graph_embedding,  # 图池化特征
            trans_embedding,  # Transformer处理后的全局特征
            freq_features  # 频域特征
        ], dim=1)  # [batch_size, hidden_dim*3]

        # 特征融合
        fused_features = self.fusion_layer(combined_features)  # [batch_size, hidden_dim]

        # 12. 最终预测
        output = self.prediction_head(fused_features)  # [batch_size, 1]

        return output

# 改进的平衡损失函数，更好地处理峰值
class EnhancedBalancedLoss(nn.Module):
    def __init__(self, mse_weight=0.5, mae_weight=0.3, peak_weight=0.2, huber_weight=0.2, r2_weight=0.1,
                 peak_threshold=0.7, huber_delta=0.1):
        super(EnhancedBalancedLoss, self).__init__()
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self.peak_weight = peak_weight
        self.huber_weight = huber_weight
        self.r2_weight = r2_weight
        self.peak_threshold = peak_threshold
        self.huber_delta = huber_delta
        self.mse = nn.MSELoss(reduction='none')
        self.mae = nn.L1Loss(reduction='none')
        self.huber = nn.SmoothL1Loss(reduction='none', beta=huber_delta)

    def forward(self, y_pred, y_true):
        # 基础损失
        mse_loss = torch.mean(self.mse(y_pred, y_true))
        mae_loss = torch.mean(self.mae(y_pred, y_true))
        huber_loss = torch.mean(self.huber(y_pred, y_true))

        # 自适应峰值检测
        with torch.no_grad():
            y_abs = torch.abs(y_true)
            batch_max = torch.max(y_abs)
            batch_std = torch.std(y_abs)
            # 动态阈值：最大值的一定比例或均值+n*标准差
            dynamic_threshold = max(
                self.peak_threshold * batch_max,
                torch.mean(y_abs) + 1.5 * batch_std
            )
            peak_mask = (y_abs > dynamic_threshold).float()
            peak_count = torch.sum(peak_mask)

        # 峰值损失
        if peak_count > 0:
            # 对峰值区域使用更高权重的MSE和MAE组合
            peak_mse = torch.sum(self.mse(y_pred, y_true) * peak_mask) / peak_count
            peak_mae = torch.sum(self.mae(y_pred, y_true) * peak_mask) / peak_count
            peak_loss = 0.7 * peak_mse + 0.3 * peak_mae
        else:
            peak_loss = 0.0

        # 添加R²损失分量（促进更好的拟合）
        if y_true.numel() > 1:  # 确保有足够的样本计算R²
            with torch.no_grad():
                y_mean = torch.mean(y_true)
                ss_tot = torch.sum((y_true - y_mean) ** 2)
                if ss_tot > 0:  # 避免除零
                    ss_res = torch.sum((y_true - y_pred) ** 2)
                    r2 = 1 - ss_res / ss_tot
                    # 转换为损失（1-R²）
                    r2_loss = 1 - r2
                else:
                    r2_loss = 0.0
        else:
            r2_loss = 0.0

        # 组合损失
        total_loss = (
                self.mse_weight * mse_loss +
                self.mae_weight * mae_loss +
                self.peak_weight * peak_loss +
                self.huber_weight * huber_loss +
                self.r2_weight * r2_loss
        )

        return total_loss

# 优化的数据预处理函数
def load_and_preprocess_data(features_path, target_path, window_size=64, stride=8):
    """增强的数据预处理函数"""
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

# 增强的数据增强函数
def enhanced_augment_data(X, y, augmentation_factor=2, noise_levels=[0.01, 0.02]):
    """高级数据增强，综合多种技术"""
    print("执行数据增强...")
    # 确保y是一维数组进行处理
    y = y.reshape(-1)

    X_aug, y_aug = [], []
    orig_samples = len(X)

    # 记录原始数据
    X_aug.append(X)
    y_aug.append(y.reshape(-1, 1))

    # 1. 添加高斯噪声 (多种噪声水平)
    for noise_level in noise_levels:
        noise = np.random.normal(0, noise_level, X.shape)
        X_noise = X + noise
        X_aug.append(X_noise)
        y_aug.append(y.reshape(-1, 1))

    # 2. 时间扭曲 (time warping)
    X_warped = []
    for i in range(len(X)):
        seq = X[i]
        # 创建一个轻微扭曲的索引
        indices = np.linspace(0, len(seq) - 1, len(seq))
        # 添加正弦波扰动
        indices = indices + np.sin(np.linspace(0, 2 * np.pi, len(seq))) * 1.5
        indices = np.clip(indices, 0, len(seq) - 1).astype(int)
        warped_seq = seq[indices]
        X_warped.append(warped_seq)

    X_aug.append(np.array(X_warped))
    y_aug.append(y.reshape(-1, 1))

    # 3. 振幅缩放
    scale_factors = [0.9, 1.1]  # 缩小和放大
    for scale in scale_factors:
        X_scaled = X * scale
        X_aug.append(X_scaled)
        y_aug.append(y.reshape(-1, 1))

    # 4. 随机窗口颠倒
    X_flipped = X.copy()
    window_size = X.shape[1] // 4
    for i in range(len(X)):
        if np.random.random() > 0.5:
            # 随机选择一个窗口
            start_idx = np.random.randint(0, X.shape[1] - window_size)
            # 颠倒窗口
            X_flipped[i, start_idx:start_idx + window_size] = X_flipped[i, start_idx:start_idx + window_size][::-1]

    X_aug.append(X_flipped)
    y_aug.append(y.reshape(-1, 1))

    # 5. 频域增强
    X_freq = []
    for i in range(len(X)):
        seq = X[i]
        # 转换到频域
        fft = np.fft.rfft(seq)
        # 随机缩放某些频率分量
        mask = np.random.uniform(0.9, 1.1, size=len(fft))
        fft_modified = fft * mask
        # 转换回时域
        seq_modified = np.fft.irfft(fft_modified, n=len(seq))
        X_freq.append(seq_modified)

    X_aug.append(np.array(X_freq))
    y_aug.append(y.reshape(-1, 1))

    # 合并所有增强数据
    X_combined = np.vstack(X_aug)
    y_combined = np.vstack(y_aug).reshape(-1)

    # 随机打乱
    indices = np.random.permutation(len(X_combined))
    X_combined = X_combined[indices]
    y_combined = y_combined[indices]

    print(f"数据增强: {orig_samples} -> {len(X_combined)} 样本")

    return X_combined, y_combined

# 定义用于将批次数据转换为PyG批次的函数
def collate_fn(batch):
    return Batch.from_data_list(batch)

# 训练函数
def train_model(model, train_loader, val_loader, num_epochs, device, patience=30, lr=3e-4):
    """增强的训练函数"""
    criterion = EnhancedBalancedLoss(
        mse_weight=0.5, mae_weight=0.3, peak_weight=0.2,
        huber_weight=0.2, r2_weight=0.1, peak_threshold=0.7
    )

    # 使用AdamW优化器
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # 学习率调度器
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr * 10,
        total_steps=num_epochs * len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    no_improve = 0

    # 保存学习率历史
    lr_history = []

    epoch_pbar = trange(num_epochs, desc="训练中")

    for epoch in epoch_pbar:
        # 训练阶段
        model.train()
        train_loss = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [训练]", leave=False)
        for batch_idx, batch in enumerate(train_pbar):
            batch = batch.to(device)
            optimizer.zero_grad()

            outputs = model(batch)
            loss = criterion(outputs, batch.y)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            # 记录当前学习率
            lr_history.append(optimizer.param_groups[0]['lr'])

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

    # 绘制学习率变化曲线
    plt.figure(figsize=(10, 4))
    plt.plot(lr_history)
    plt.title('学习率调度曲线')
    plt.xlabel('训练步数')
    plt.ylabel('学习率')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('lr_schedule.png')

    return train_losses, val_losses

# 评估函数
def evaluate_model(model, test_loader, scaler_target, device):
    """评估模型函数"""
    model.eval()
    predictions = []
    actuals = []

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

    # 计算峰值区域的指标
    abs_actuals = np.abs(actuals)
    peak_threshold = 0.7 * np.max(abs_actuals)
    peak_indices = np.where(abs_actuals > peak_threshold)[0]

    if len(peak_indices) > 0:
        peak_r2 = r2_score(actuals[peak_indices], predictions[peak_indices])
        peak_rmse = np.sqrt(mean_squared_error(actuals[peak_indices], predictions[peak_indices]))
        peak_mae = mean_absolute_error(actuals[peak_indices], predictions[peak_indices])
    else:
        peak_r2, peak_rmse, peak_mae = 0, 0, 0

    return {
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae,
        'Peak_R2': peak_r2,
        'Peak_RMSE': peak_rmse,
        'Peak_MAE': peak_mae,
        'predictions': predictions,
        'actuals': actuals
    }

# 扩展可视化函数
def visualize_results(results, train_losses, val_losses):
    """增强的结果可视化函数"""
    # 创建一个更大的图形
    plt.figure(figsize=(20, 15))

    # 1. 预测对比图 - 前500个样本
    plt.subplot(3, 2, 1)
    plt.plot(results['actuals'][:500], label='真实值', color='navy', alpha=0.7)
    plt.plot(results['predictions'][:500], label='预测值', color='crimson', alpha=0.7)
    plt.legend(fontsize=12)
    plt.title(f'预测结果对比 (前500个样本)\nR² = {results["R2"]:.4f}', fontsize=14)
    plt.xlabel('样本索引', fontsize=12)
    plt.ylabel('值', fontsize=12)
    plt.grid(True, alpha=0.3)

    # 2. 预测对比图 - 中间500个样本
    middle_idx = len(results['actuals']) // 2
    plt.subplot(3, 2, 2)
    plt.plot(results['actuals'][middle_idx:middle_idx + 500], label='真实值', color='navy', alpha=0.7)
    plt.plot(results['predictions'][middle_idx:middle_idx + 500], label='预测值', color='crimson', alpha=0.7)
    plt.legend(fontsize=12)
    plt.title(f'预测结果对比 (中间500个样本)\nR² = {results["R2"]:.4f}', fontsize=14)
    plt.xlabel('样本索引', fontsize=12)
    plt.ylabel('值', fontsize=12)
    plt.grid(True, alpha=0.3)

    # 3. 损失曲线
    plt.subplot(3, 2, 3)
    plt.plot(train_losses, label='训练损失', color='blue', alpha=0.7)
    plt.plot(val_losses, label='验证损失', color='red', alpha=0.7)
    plt.legend(fontsize=12)
    plt.title('训练过程损失曲线', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)

    # 4. 真实值vs预测值的散点图
    plt.subplot(3, 2, 4)
    plt.scatter(results['actuals'], results['predictions'], alpha=0.1, s=1, color='blue')
    min_val = min(np.min(results['actuals']), np.min(results['predictions']))
    max_val = max(np.max(results['actuals']), np.max(results['predictions']))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='理想预测线')
    plt.legend(fontsize=12)
    plt.title(f'真实值 vs 预测值\nR² = {results["R2"]:.4f}', fontsize=14)
    plt.xlabel('真实值', fontsize=12)
    plt.ylabel('预测值', fontsize=12)
    plt.grid(True, alpha=0.3)

    # 5. 绝对误差分布
    plt.subplot(3, 2, 5)
    errors = np.abs(results['predictions'] - results['actuals'])
    plt.hist(errors, bins=50, alpha=0.7, color='green')
    plt.axvline(np.mean(errors), color='red', linestyle='--', label=f'平均误差: {np.mean(errors):.4f}')
    plt.legend(fontsize=12)
    plt.title('预测绝对误差分布', fontsize=14)
    plt.xlabel('绝对误差', fontsize=12)
    plt.ylabel('频率', fontsize=12)
    plt.grid(True, alpha=0.3)

    # 6. 最后500个样本
    plt.subplot(3, 2, 6)
    plt.plot(results['actuals'][-500:], label='真实值', color='navy', alpha=0.7)
    plt.plot(results['predictions'][-500:], label='预测值', color='crimson', alpha=0.7)
    plt.legend(fontsize=12)
    plt.title(f'预测结果对比 (最后500个样本)\nR² = {results["R2"]:.4f}', fontsize=14)
    plt.xlabel('样本索引', fontsize=12)
    plt.ylabel('值', fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('model_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 峰值区域的放大图
    plt.figure(figsize=(15, 10))

    # 寻找最大波动区域
    rolling_std = pd.Series(results['actuals']).rolling(50).std()
    if not rolling_std.isna().all():  # 确保有有效值
        volatile_idx = rolling_std.idxmax()
        start_idx = max(0, volatile_idx - 100)
        end_idx = min(len(results['actuals']), volatile_idx + 100)

        plt.subplot(2, 1, 1)
        plt.plot(range(start_idx, end_idx), results['actuals'][start_idx:end_idx],
                 label='真实值', color='navy', linewidth=2)
        plt.plot(range(start_idx, end_idx), results['predictions'][start_idx:end_idx],
                 label='预测值', color='crimson', linewidth=2)
        plt.legend(fontsize=14)
        plt.title(f'高波动区域预测对比 (样本 {start_idx}-{end_idx})\n峰值区域 R² = {results["Peak_R2"]:.4f}',
                  fontsize=16)
        plt.xlabel('样本索引', fontsize=14)
        plt.ylabel('值', fontsize=14)
        plt.grid(True, alpha=0.3)

        # 显示局部误差
        plt.subplot(2, 1, 2)
        local_errors = results['predictions'][start_idx:end_idx] - results['actuals'][start_idx:end_idx]
        plt.bar(range(start_idx, end_idx), local_errors, alpha=0.5, color='green')
        plt.axhline(y=0, color='red', linestyle='-', alpha=0.7)
        plt.title('局部预测误差', fontsize=16)
        plt.xlabel('样本索引', fontsize=14)
        plt.ylabel('误差 (预测 - 真实)', fontsize=14)
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('peak_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # 设置参数
    WINDOW_SIZE = 64
    STRIDE = 4  # 减小步长以获取更多样本
    BATCH_SIZE = 128
    NUM_EPOCHS = 200
    LEARNING_RATE = 3e-4

    print("开始数据预处理...")
    X, y, scaler_features, scaler_target = load_and_preprocess_data(
        'xtrain_new.csv',
        'ytrain_new.csv',
        WINDOW_SIZE,
        STRIDE
    )

    print("执行增强数据增强...")
    X_augmented, y_augmented = enhanced_augment_data(X, y, augmentation_factor=2)
    print(f"增强后的数据形状 - X: {X_augmented.shape}, y: {y_augmented.shape}")

    print("划分数据集...")
    # 使用分层抽样保证训练集和测试集分布一致
    X_train, X_test, y_train, y_test = train_test_split(
        X_augmented, y_augmented, test_size=0.15, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42
    )

    print("创建图数据集...")
    train_dataset = SeismicGraphDataset(X_train, y_train, window_size=WINDOW_SIZE, k_neighbors=12,
                                        time_decay=0.05, include_spectral=True)
    val_dataset = SeismicGraphDataset(X_val, y_val, window_size=WINDOW_SIZE, k_neighbors=12,
                                      time_decay=0.05, include_spectral=True)
    test_dataset = SeismicGraphDataset(X_test, y_test, window_size=WINDOW_SIZE, k_neighbors=12,
                                       time_decay=0.05, include_spectral=True)

    # 根据系统设置worker数
    num_workers = 0  # 如果有多核CPU，可以设置为4或更高

    print("创建数据加载器...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    print("初始化增强型图神经网络模型...")
    freq_dim = WINDOW_SIZE // 2 + 1  # FFT结果的维度

    model = EnhancedSTGraphNet(
        node_feature_dim=10,  # 增强的节点特征: 值+位置编码+三角位置+3个局部差分+3个局部统计
        hidden_dim=192,
        freq_dim=freq_dim,  # 频域特征维度
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
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, NUM_EPOCHS, device, patience=30, lr=LEARNING_RATE
    )

    print("加载最佳模型...")
    model.load_state_dict(torch.load('best_model.pth'))

    # 评估模型
    print("评估模型...")
    results = evaluate_model(model, test_loader, scaler_target, device)

    print("\n=== 增强型图神经网络模型评估结果 ===")
    print(f"R2 分数: {results['R2']:.4f}")
    print(f"RMSE: {results['RMSE']:.4f}")
    print(f"MAE: {results['MAE']:.4f}")
    print(f"峰值区域 R2: {results['Peak_R2']:.4f}")
    print(f"峰值区域 RMSE: {results['Peak_RMSE']:.4f}")
    print(f"峰值区域 MAE: {results['Peak_MAE']:.4f}")

    # 绘制结果
    print("\n绘制结果...")
    visualize_results(results, train_losses, val_losses)

    print("\n训练完成!")

if __name__ == '__main__':
    main()