import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tqdm import tqdm
import os
import json
import argparse
import sys
from pathlib import Path
import datetime
import re

# ==================== 配置参数 ====================
def parse_args():
    parser = argparse.ArgumentParser(description='DeepPlantTE模型预测脚本 - 支持CSV和FASTA格式')
    
    # 必需参数
    parser.add_argument('--config', type=str, required=True,
                        help='配置文件路径(config.json) - 必需')
    parser.add_argument('--model', type=str, required=True,
                        help='模型权重文件路径(best_DeepPlantTE_model.pth) - 必需')
    parser.add_argument('--label_encoder', type=str, required=True,
                        help='标签编码器JSON文件路径(label_encoder.json) - 必需')
    
    # 输入文件相关（二选一）
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input_csv', type=str, help='输入CSV文件路径')
    input_group.add_argument('--input_fasta', type=str, help='输入FASTA文件路径')
    
    # CSV相关参数
    parser.add_argument('--seq_col', type=str, default=None,
                        help='CSV文件中的序列列名(当使用--input_csv时必须)')
    parser.add_argument('--label_col', type=str, default=None,
                        help='CSV文件中的标签列名(可选，如果提供会在输出中保留)')
    
    # FASTA相关参数
    parser.add_argument('--output_fasta', type=str, default=None,
                        help='输出FASTA文件路径(当使用--input_fasta时推荐)')
    parser.add_argument('--top_k', type=int, default=3,
                        help='在FASTA头信息中显示top-k个预测类别(默认: 3)')
    parser.add_argument('--min_prob', type=float, default=0.01,
                        help='在FASTA头信息中显示的最小概率阈值(默认: 0.01)')
    
    # 通用参数
    parser.add_argument('--batch_size', type=int, default=None,
                        help='批次大小(如果不指定，将从config读取)')
    parser.add_argument('--max_seq_len', type=int, default=None,
                        help='序列长度(如果不指定，将从config读取)')
    parser.add_argument('--device', type=str, default=None,
                        help='设备(cuda/cpu)，默认自动选择')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录(如果不指定，将在代码目录创建Predict_DeepPlantTE_时间戳文件夹)')
    parser.add_argument('--output_prefix', type=str, default='predictions',
                        help='输出文件前缀(默认: predictions)')
    parser.add_argument('--output_csv', action='store_true', default=False,
                        help='同时输出CSV格式的预测结果(包含详细概率)')
    
    return parser.parse_args()

# ==================== 设置随机种子 ====================
def set_seed(seed):
    """设置随机种子"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==================== 标签编码器加载函数 ====================
def load_label_encoder(json_path):
    """从JSON文件加载LabelEncoder"""
    with open(json_path, 'r', encoding='utf-8') as f:
        label_mapping = json.load(f)
    
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(label_mapping['classes'])
    return label_encoder

# ==================== FASTA解析器 ====================
def parse_fasta(file_path):
    """解析FASTA文件，返回header列表和序列列表"""
    headers = []
    sequences = []
    current_header = None
    current_seq = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_header is not None:
                    headers.append(current_header)
                    sequences.append(''.join(current_seq))
                current_header = line[1:]  # 去掉'>'
                current_seq = []
            else:
                current_seq.append(line)
        
        # 处理最后一个序列
        if current_header is not None:
            headers.append(current_header)
            sequences.append(''.join(current_seq))
    
    return headers, sequences

def write_fasta_with_predictions(input_fasta, output_fasta, headers, predictions, probs, class_names, top_k=3, min_prob=0.01):
    """将预测结果写入新的FASTA文件，在原头信息后添加预测信息"""
    with open(input_fasta, 'r', encoding='utf-8') as fin, \
         open(output_fasta, 'w', encoding='utf-8') as fout:
        
        seq_idx = -1
        for line in fin:
            line = line.rstrip()
            if line.startswith('>'):
                seq_idx += 1
                header = line[1:]  # 去掉'>'
                
                # 构建预测信息字符串
                pred_class = class_names[predictions[seq_idx]]
                pred_prob = probs[seq_idx][predictions[seq_idx]]
                
                # 获取top-k预测
                top_indices = np.argsort(probs[seq_idx])[::-1][:top_k]
                top_info = []
                for idx in top_indices:
                    prob = probs[seq_idx][idx]
                    if prob >= min_prob:
                        top_info.append(f"{class_names[idx]}:{prob:.4f}")
                
                # 组合新的header
                if top_info:
                    new_header = f">{header} #predicted_class={pred_class} confidence={pred_prob:.4f} top{len(top_info)}={','.join(top_info)}"
                else:
                    new_header = f">{header} #predicted_class={pred_class} confidence={pred_prob:.4f}"
                
                fout.write(new_header + '\n')
            else:
                fout.write(line + '\n')

# ==================== DeepPlantTE模型定义 ====================
class DNAEmbedding(nn.Module):
    """DNA序列嵌入层"""
    def __init__(self, vocab_size=5, embed_dim=128, max_len=2000):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_len = max_len
        
        # 1. 基础投影层
        self.base_projection = nn.Sequential(
            nn.Linear(vocab_size, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # 2. 位置编码（可学习）
        self.position_encoding = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)
        
        # 3. 化学性质感知卷积（捕捉局部模式）
        self.chem_conv = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2, groups=4),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Dropout(0.05)
        )
        
        # 4. 层归一化
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # 5. 最终dropout
        self.final_dropout = nn.Dropout(0.1)
        
    def forward(self, x_onehot):
        """
        x_onehot: [batch_size, seq_len, vocab_size]
        返回: [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = x_onehot.shape
        
        # 重塑以适用于BatchNorm1d
        x_flat = x_onehot.reshape(-1, self.vocab_size)
        base_emb = self.base_projection(x_flat)
        base_emb = base_emb.reshape(batch_size, seq_len, self.embed_dim)
        
        # 添加位置编码
        pos_emb = self.position_encoding[:, :seq_len, :]
        emb = base_emb + pos_emb
        
        # 化学性质感知卷积
        emb_transposed = emb.transpose(1, 2)
        chem_enhanced = self.chem_conv(emb_transposed)
        chem_enhanced = chem_enhanced.transpose(1, 2)
        
        # 残差连接 + 层归一化
        emb = emb + chem_enhanced
        emb = self.layer_norm(emb)
        
        return self.final_dropout(emb)


class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.activation1 = nn.GELU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(channels)
        self.activation2 = nn.GELU()
    
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + residual
        x = self.activation2(x)
        return x


class LKA1D(nn.Module):
    """大核注意力模块"""
    def __init__(self, dim):
        super().__init__()
        # 5x1卷积，用于局部特征提取
        self.conv0 = nn.Conv1d(dim, dim, kernel_size=5, padding=2, groups=dim)
        # 7x1空洞卷积，用于扩大感受野
        self.conv_spatial = nn.Conv1d(dim, dim, kernel_size=7, stride=1, padding=9, 
                                     groups=dim, dilation=3)
        # 1x1卷积，用于特征融合和通道调整
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=1)
        
    def forward(self, x):
        # 保留原始输入
        u = x.clone()
        # 大核注意力计算
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        # 残差连接 + 注意力加权
        return u * attn


class MultiScaleCNN(nn.Module):
    """DeepPlantTE多尺度CNN模块"""
    def __init__(self, in_channels, out_channels_per_kernel=64):
        super(MultiScaleCNN, self).__init__()
        
        # 瓶颈层
        self.bottleneck = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=1, padding=0),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # 多尺度卷积核
        kernel_sizes = [3, 5, 7, 9, 11, 13, 15]
        self.convs = nn.ModuleList()
        
        for kernel_size in kernel_sizes:
            conv = nn.Sequential(
                nn.Conv1d(64, out_channels_per_kernel, kernel_size=kernel_size, 
                         padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels_per_kernel),
                nn.GELU(),
                ResidualBlock(out_channels_per_kernel, kernel_size=3),
                nn.Dropout(0.15)
            )
            self.convs.append(conv)
        
        # 输出通道数
        self.out_channels = len(kernel_sizes) * out_channels_per_kernel
    
    def forward(self, x):
        # 通过瓶颈层
        x = self.bottleneck(x)
        
        conv_outputs = []
        for conv in self.convs:
            out = conv(x)
            conv_outputs.append(out)
        
        out = torch.cat(conv_outputs, dim=1)
        return out


class DeepPlantTE(nn.Module):
    """DeepPlantTE模型 """
    def __init__(self, vocab_size=5, embed_dim=128, hidden_dim=320, num_classes=10, 
                 num_lstm_layers=2, max_seq_len=2000):
        super(DeepPlantTE, self).__init__()
        
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # ==================== 嵌入层 ====================
        self.embedding = DNAEmbedding(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            max_len=max_seq_len
        )
        
        # ==================== CNN-BiLSTM分支 ====================
        self.multi_scale_cnn = MultiScaleCNN(embed_dim, out_channels_per_kernel=64)
        cnn_output_dim = self.multi_scale_cnn.out_channels  
        
        # ==================== LKA-1D注意力模块 ====================
        self.lka = LKA1D(cnn_output_dim)
        
        self.cnn_dropout = nn.Dropout(0.15)
        
        self.bilstm = nn.LSTM(
            cnn_output_dim, hidden_dim // 2, num_layers=num_lstm_layers,
            bidirectional=True, batch_first=True, 
            dropout=0.15 if num_lstm_layers > 1 else 0
        )
        self.bilstm_dropout = nn.Dropout(0.15)
        
        self.attention_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # ==================== 特征融合 ====================
        fusion_input_dim = hidden_dim
        
        self.fusion_projection = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim * 2), 
            nn.BatchNorm1d(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # ==================== 特征头 ====================
        self.feature_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),  
            nn.BatchNorm1d(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim * 2, hidden_dim),  
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        
        # ==================== 分类头 ====================
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), 
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),  
            nn.BatchNorm1d(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(hidden_dim // 4, num_classes)  
        )
    
    def apply_attention_mask(self, attn_weights, seq_lengths, max_len):
        batch_size = attn_weights.shape[0]
        mask = torch.arange(max_len, device=attn_weights.device).unsqueeze(0) < seq_lengths.unsqueeze(1)
        
        if attn_weights.dtype == torch.float16:
            mask_value = torch.tensor(-65500.0, dtype=torch.float16, device=attn_weights.device)
        else:
            mask_value = torch.tensor(-1e9, dtype=torch.ones_like(attn_weights).dtype, device=attn_weights.device)
        
        attn_weights = attn_weights.masked_fill(~mask, mask_value)
        return attn_weights
    
    def forward(self, x_full, full_seq_len=None):
        batch_size, seq_len, vocab_size = x_full.shape
        
        # ==================== 嵌入处理 ====================
        embedded = self.embedding(x_full)
        
        # 转置以适配CNN
        embedded_t = embedded.transpose(1, 2)
        
        # CNN多尺度特征提取
        cnn_features = self.multi_scale_cnn(embedded_t)
        
        # 应用LKA-1D大核注意力
        cnn_features = self.lka(cnn_features)
        
        cnn_features = self.cnn_dropout(cnn_features)
        cnn_features_t = cnn_features.transpose(1, 2)
        
        # CNN-BiLSTM时序建模
        bilstm_out, _ = self.bilstm(cnn_features_t)
        attn_weights = self.attention_gate(bilstm_out)
        attn_weights = attn_weights.squeeze(-1)
        
        if full_seq_len is not None:
            attn_weights = self.apply_attention_mask(
                attn_weights, full_seq_len, bilstm_out.shape[1]
            )
        
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(bilstm_out * attn_weights.unsqueeze(-1), dim=1)
        context = self.bilstm_dropout(context)
        
        # ==================== 特征融合 ====================
        fused_features = self.fusion_projection(context)
        
        # ==================== 分类 ====================
        features = self.feature_head(fused_features)
        logits = self.classifier(features)
        
        return logits, features


# ==================== 数据集类 ====================
class DeepPlantTEPredictDataset(Dataset):
    """DeepPlantTE预测数据集类 - 支持从序列列表创建"""
    def __init__(self, sequences, max_seq_len, indices=None, df=None):
        self.sequences = sequences
        self.max_seq_len = max_seq_len
        self.indices = indices if indices is not None else np.arange(len(sequences))
        self.df = df  # 用于CSV模式保存原始数据
        
        self.char_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        self.vocab_size = len(self.char_to_idx)
    
    def __len__(self):
        return len(self.sequences)
    
    def sequence_to_onehot(self, seq, max_len):
        """将序列转换为one-hot编码"""
        seq_encoded = [self.char_to_idx.get(c, self.char_to_idx['N']) for c in seq]
        
        # 截断或填充
        if len(seq_encoded) > max_len:
            seq_encoded = seq_encoded[:max_len]
        else:
            seq_encoded += [self.char_to_idx['N']] * (max_len - len(seq_encoded))
        
        onehot = np.zeros((max_len, self.vocab_size), dtype=np.float32)
        for i, idx in enumerate(seq_encoded):
            onehot[i, idx] = 1.0
        
        return onehot
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        full_encoded = self.sequence_to_onehot(seq, self.max_seq_len)
        
        return (torch.tensor(full_encoded, dtype=torch.float32),
                torch.tensor(len(seq), dtype=torch.long),
                self.indices[idx])  # 返回索引，用于对齐数据


# ==================== 预测函数 ====================
def predict(model, dataloader, device):
    """预测函数 - 只返回预测结果和概率"""
    model.eval()
    all_preds = []
    all_probs = []
    all_indices = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            sequences_full, full_seq_len, indices = batch
            full_seq_len = full_seq_len.to(device)
            sequences_full = sequences_full.to(device)
            
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    logits, _ = model(sequences_full, full_seq_len)
            else:
                logits, _ = model(sequences_full, full_seq_len)
            
            probs = F.softmax(logits, dim=1)
            _, predicted = torch.max(logits, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_indices.extend(indices.numpy())
    
    # 按原始顺序排序
    sorted_indices = np.argsort(all_indices)
    all_preds = np.array(all_preds)[sorted_indices]
    all_probs = np.array(all_probs)[sorted_indices]
    
    return all_preds, all_probs


# ==================== CSV保存函数 ====================
def save_csv_predictions(dataset, all_preds, all_probs, class_names, 
                         output_dir, output_prefix, label_col=None):
    """保存CSV格式的预测结果"""
    
    # 创建结果DataFrame
    results_df = dataset.df.copy()
    
    # 添加预测结果
    pred_labels_names = [class_names[pred] for pred in all_preds]
    results_df['predicted_label'] = pred_labels_names
    results_df['predicted_label_numeric'] = all_preds
    
    # 添加每个类别的概率
    for i, class_name in enumerate(class_names):
        results_df[f'prob_{class_name}'] = all_probs[:, i]
    
    # 如果提供了标签列，保留原始标签；否则删除标签列（如果存在）
    if label_col is None:
        # 删除可能的标签列
        label_columns = [col for col in results_df.columns 
                        if col.endswith('_label') or col.endswith('_Category') or col == 'Mapped_Category2']
        for col in label_columns:
            if col in results_df.columns:
                results_df = results_df.drop(columns=[col])
    
    # 保存CSV
    output_path = os.path.join(output_dir, f'{output_prefix}.csv')
    results_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"CSV预测结果已保存到: {output_path}")
    
    return results_df


# ==================== 统计函数 ====================
def print_statistics(all_preds, class_names):
    """打印预测统计信息"""
    total_sequences = len(all_preds)
    unique, counts = np.unique(all_preds, return_counts=True)
    
    print(f"\n{'='*60}")
    print("预测统计")
    print(f"{'='*60}")
    print(f"总共预测序列数: {total_sequences}")
    print(f"\n各类别预测数量:")
    print(f"{'-'*40}")
    
    # 创建类别计数字典
    pred_counts = {}
    for i, class_name in enumerate(class_names):
        pred_counts[class_name] = 0
    
    for pred_idx, count in zip(unique, counts):
        class_name = class_names[pred_idx]
        pred_counts[class_name] = count
    
    # 按类别名称排序输出
    for class_name in class_names:
        count = pred_counts[class_name]
        percentage = (count / total_sequences) * 100 if total_sequences > 0 else 0
        print(f"  {class_name:30s}: {count:6d} ({percentage:5.2f}%)")
    
    print(f"{'='*60}\n")
    
    return pred_counts


# ==================== 主函数 ====================
def main():
    # 打印pandas版本信息用于调试
    print(f"Pandas版本: {pd.__version__}")
    print(f"Pandas导入成功!")
    
    args = parse_args()
    
    # 检查必需参数
    required_args = ['config', 'model', 'label_encoder']
    missing_args = [arg for arg in required_args if not getattr(args, arg)]
    if missing_args:
        print(f"错误: 缺少必需参数: {', '.join(missing_args)}")
        sys.exit(1)
    
    # 检查输入文件相关参数
    if args.input_csv and not args.seq_col:
        print("错误: 使用--input_csv时必须指定--seq_col")
        sys.exit(1)
    
    # 检查文件是否存在
    files_to_check = [('config', args.config), 
                      ('model', args.model), 
                      ('label_encoder', args.label_encoder)]
    
    if args.input_csv:
        files_to_check.append(('input_csv', args.input_csv))
    if args.input_fasta:
        files_to_check.append(('input_fasta', args.input_fasta))
    
    for arg_name, file_path in files_to_check:
        if not os.path.exists(file_path):
            print(f"错误: {arg_name}文件不存在: {file_path}")
            sys.exit(1)
    
    # 设置设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 设置输出目录
    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
    else:
        # 在代码目录创建Predict_DeepPlantTE_时间戳文件夹
        script_dir = os.path.dirname(os.path.abspath(__file__))
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(script_dir, f"Predict_DeepPlantTE_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    # ==================== 加载配置 ====================
    config_path = os.path.abspath(args.config)
    print(f"加载配置: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # ==================== 加载标签编码器 ====================
    label_encoder_path = os.path.abspath(args.label_encoder)
    print(f"加载标签编码器: {label_encoder_path}")
    label_encoder = load_label_encoder(label_encoder_path)
    class_names = label_encoder.classes_.tolist()
    num_classes = len(class_names)
    print(f"类别名称: {class_names}")
    
    # ==================== 设置随机种子 ====================
    set_seed(config.get('seed', 42))
    
    # ==================== 确定输入数据 ====================
    sequences = []
    df = None
    headers = None
    input_type = "CSV" if args.input_csv else "FASTA"
    
    if args.input_csv:
        input_path = os.path.abspath(args.input_csv)
        print(f"输入CSV文件: {input_path}")
        print(f"序列列名: {args.seq_col}")
        if args.label_col:
            print(f"标签列名: {args.label_col}")
        else:
            print(f"标签列名: 未指定")
        
        try:
            # 读取CSV，尝试不同的编码
            try:
                df = pd.read_csv(input_path, encoding='utf-8')
            except UnicodeDecodeError:
                print("UTF-8编码失败，尝试GBK编码...")
                df = pd.read_csv(input_path, encoding='gbk')
            except Exception as e:
                print(f"读取CSV文件时出错: {e}")
                sys.exit(1)
            
            # 检查序列列是否存在
            if args.seq_col not in df.columns:
                print(f"错误: 在CSV文件中找不到序列列 '{args.seq_col}'")
                print(f"可用的列: {list(df.columns)}")
                sys.exit(1)
            
            sequences = df[args.seq_col].values.tolist()
            print(f"成功读取 {len(sequences)} 条序列")
            
        except Exception as e:
            print(f"读取CSV文件时发生错误: {e}")
            sys.exit(1)
        
    else:  # FASTA
        input_path = os.path.abspath(args.input_fasta)
        print(f"输入FASTA文件: {input_path}")
        try:
            headers, sequences = parse_fasta(input_path)
            print(f"FASTA文件包含 {len(headers)} 条序列")
        except Exception as e:
            print(f"读取FASTA文件时发生错误: {e}")
            sys.exit(1)
        
        if not args.output_fasta:
            args.output_fasta = os.path.join(output_dir, f"{args.output_prefix}.fa")
            print(f"输出FASTA文件: {args.output_fasta}")
    
    # ==================== 确定批次大小 ====================
    batch_size = args.batch_size if args.batch_size else config.get('batch_size', 128)
    
    # ==================== 确定序列长度 ====================
    # 优先使用命令行参数
    max_seq_len = args.max_seq_len
    
    # 如果命令行没指定，从配置读取
    if max_seq_len is None:
        max_seq_len = config.get('actual_max_seq_len')
        if max_seq_len is None:
            max_seq_len = config.get('max_seq_len_percentile')
            if max_seq_len is None:
                print("错误: 配置文件中找不到序列长度信息 (actual_max_seq_len 或 max_seq_len_percentile)")
                sys.exit(1)
            
            # 如果是百分位数，转换为具体数值
            if isinstance(max_seq_len, float) and max_seq_len < 100:
                # 从输入数据计算百分位数
                seq_lengths = [len(seq) for seq in sequences]
                max_seq_len = int(np.percentile(seq_lengths, max_seq_len))
                print(f"从输入数据计算{config.get('max_seq_len_percentile', 90)}%分位数: {max_seq_len}")
    
    print(f"\n预测配置:")
    print(f"  输入类型: {input_type}")
    print(f"  输入文件: {input_path}")
    print(f"  序列数量: {len(sequences)}")
    print(f"  批次大小: {batch_size}")
    print(f"  序列长度: {max_seq_len}")
    print(f"  类别数量: {num_classes}")
    
    # ==================== 创建预测数据集 ====================
    predict_dataset = DeepPlantTEPredictDataset(
        sequences=sequences,
        max_seq_len=max_seq_len,
        df=df  # 传入df，用于CSV模式
    )
    
    predict_loader = DataLoader(
        predict_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # ==================== 加载模型 ====================
    model_path = os.path.abspath(args.model)
    print(f"\n加载模型: {model_path}")
    
    # 创建模型
    model = DeepPlantTE(
        vocab_size=5,
        embed_dim=config.get('embed_dim', 128),
        hidden_dim=config.get('hidden_dim', 320),
        num_classes=num_classes,
        num_lstm_layers=config.get('num_lstm_layers', 2),
        max_seq_len=max_seq_len + 100  # 当前数据集仍按原始 max_seq_len 截断，因此这100个位置并未实际使用。 保留此设置仅为防止未来修改代码时出现越界，属于防御性冗余
    ).to(device)
    
    try:
        # 先尝试直接加载
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("模型加载成功!")
    except Exception as e:
        try:
            # 如果失败，尝试去掉module前缀（如果是DataParallel保存的）
            print(f"尝试使用DataParallel格式加载...")
            state_dict = torch.load(model_path, map_location=device)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict)
            print("模型加载成功!")
        except Exception as e:
            print(f"加载模型失败: {e}")
            sys.exit(1)
    
    model.eval()
    
    # ==================== 运行预测 ====================
    print(f"\n开始预测...")
    all_preds, all_probs = predict(model, predict_loader, device)
    
    # ==================== 打印统计信息 ====================
    pred_counts = print_statistics(all_preds, class_names)
    
    # ==================== 保存结果 ====================
    if args.input_csv:
        # CSV模式：保存CSV预测结果
        save_csv_predictions(
            predict_dataset, all_preds, all_probs, class_names,
            output_dir, args.output_prefix, args.label_col
        )
        
        # 如果需要同时输出FASTA（但CSV模式不输出FASTA）
        if args.output_fasta:
            print("警告: CSV模式下不输出FASTA文件，忽略--output_fasta")
    
    else:  # FASTA模式
        # 保存FASTA预测结果
        if args.output_fasta:
            write_fasta_with_predictions(
                input_path, args.output_fasta, headers, all_preds, all_probs, 
                class_names, args.top_k, args.min_prob
            )
            print(f"FASTA预测结果已保存到: {args.output_fasta}")
        
        # 如果需要同时输出CSV
        if args.output_csv:
            # 为FASTA模式创建虚拟的DataFrame用于CSV输出
            fasta_df = pd.DataFrame({
                'header': headers,
                'sequence': sequences
            })
            
            # 创建临时数据集用于CSV输出
            temp_dataset = DeepPlantTEPredictDataset(
                sequences=sequences,
                max_seq_len=max_seq_len,
                df=fasta_df
            )
            
            save_csv_predictions(
                temp_dataset, all_preds, all_probs, class_names,
                output_dir, f"{args.output_prefix}_details", None
            )
    
    # 保存统计信息到JSON
    stats = {
        'total_sequences': int(len(all_preds)),
        'prediction_counts': {class_name: int(pred_counts[class_name]) for class_name in class_names},
        'class_names': class_names,
        'input_file': input_path,
        'input_type': input_type,
        'batch_size': batch_size,
        'max_seq_len': max_seq_len,
        'top_k': args.top_k if args.input_fasta else None,
        'min_prob': args.min_prob if args.input_fasta else None,
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    stats_path = os.path.join(output_dir, f'{args.output_prefix}_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)
    print(f"统计信息已保存到: {stats_path}")
    
    print(f"\n预测完成! 结果保存在: {output_dir}")
    print(f"输出文件前缀: {args.output_prefix}")


if __name__ == "__main__":
    main()