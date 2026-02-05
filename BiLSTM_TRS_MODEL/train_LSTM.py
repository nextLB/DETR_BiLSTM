
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
from collections import Counter
import warnings
import random
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class VehicleTrajectoryDataset(Dataset):
    """改进的车辆轨迹数据集类，包含数据增强和更好的特征工程"""

    def __init__(self, data_dir, label_dir, max_seq_len=100, augment=False):
        """
        初始化数据集

        Args:
            data_dir: 数据文件目录
            label_dir: 标签文件目录
            max_seq_len: 最大序列长度（填充/截断）
            augment: 是否使用数据增强
        """
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.max_seq_len = max_seq_len
        self.augment = augment
        self.sequences = []
        self.labels = []
        self.sequence_lengths = []
        self.feature_scaler = StandardScaler()

        # 获取所有数据文件
        data_files = [f for f in os.listdir(data_dir) if f.endswith('_extract_data.json')]
        data_files.sort()

        all_trajectories = []
        all_labels = []
        all_lengths = []

        for data_file in tqdm(data_files, desc="Loading data files"):
            # 构建对应的标签文件名
            base_name = data_file.replace('_extract_data.json', '')
            label_file = f"{base_name}_label.txt"

            if not os.path.exists(os.path.join(label_dir, label_file)):
                continue

            # 加载数据
            data_path = os.path.join(data_dir, data_file)
            with open(data_path, 'r') as f:
                data = json.load(f)

            # 加载标签
            label_path = os.path.join(label_dir, label_file)
            with open(label_path, 'r') as f:
                label_lines = f.readlines()

            # 解析标签文件
            label_dict = {}
            for line in label_lines[2:]:
                line = line.strip()
                if line:
                    parts = line.split(',')
                    if len(parts) >= 2:
                        vehicle_id = parts[0]
                        label_str = parts[1].lower()
                        if label_str == 'rash':
                            label = 1
                        elif label_str == 'accident':
                            label = 2
                        else:
                            label = 0
                        label_dict[vehicle_id] = label

            # 处理每个车辆的轨迹
            for vehicle_id, trajectory in data.items():
                # 跳过第一个[-1,-1,-1,-1,-1]
                if len(trajectory) < 3:
                    continue

                # 提取轨迹点（跳过第一个-1数组）
                traj_points = []
                for point in trajectory[1:]:
                    if isinstance(point, list) and len(point) >= 4:
                        x1, y1, x2, y2 = point[:4]
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        width = x2 - x1
                        height = y2 - y1
                        traj_points.append([center_x, center_y, width, height])

                if len(traj_points) < 5:  # 增加最小序列长度要求
                    continue

                # 添加丰富的运动特征
                traj_with_features = self._add_rich_features(traj_points)

                # 获取标签
                label = label_dict.get(vehicle_id, 0)

                all_trajectories.append(traj_with_features)
                all_labels.append(label)
                all_lengths.append(len(traj_with_features))

                # 数据增强：生成变体
                if self.augment and len(traj_points) > 10:
                    # 1. 轻微扰动
                    perturbed = self._augment_perturb(traj_with_features.copy())
                    all_trajectories.append(perturbed)
                    all_labels.append(label)
                    all_lengths.append(len(perturbed))

                    # 2. 时间缩放
                    scaled = self._augment_temporal_scale(traj_with_features.copy())
                    all_trajectories.append(scaled)
                    all_labels.append(label)
                    all_lengths.append(len(scaled))

        # 特征归一化
        self._fit_feature_scaler(all_trajectories)

        for traj in all_trajectories:
            # 截断或填充序列
            if len(traj) > self.max_seq_len:
                traj = traj[:self.max_seq_len]
                seq_len = self.max_seq_len
            else:
                padding = [[0] * traj[0].shape[0]] * (self.max_seq_len - len(traj))
                traj.extend(padding)
                seq_len = len(traj)

            # 归一化特征
            traj_norm = self.feature_scaler.transform(traj)
            self.sequences.append(traj_norm)
            self.sequence_lengths.append(seq_len)

        self.labels = np.array(all_labels, dtype=np.int64)
        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.sequence_lengths = np.array(self.sequence_lengths, dtype=np.int64)

        print(f"\nDataset loaded: {len(self.sequences)} sequences")
        print(f"Class distribution: {Counter(self.labels)}")
        print(f"Feature dimension: {self.sequences[0].shape[1]}")

    def _fit_feature_scaler(self, trajectories):
        """拟合特征标准化器"""
        all_features = []
        for traj in trajectories:
            all_features.extend(traj)
        self.feature_scaler.fit(all_features)

    def _add_rich_features(self, traj_points):
        """添加丰富的运动特征"""
        traj_with_features = []

        for i in range(len(traj_points)):
            # 基本位置特征
            center_x, center_y, width, height = traj_points[i]

            # 计算速度
            if i == 0:
                vx, vy = 0, 0
                speed = 0
                direction = 0
            else:
                vx = traj_points[i][0] - traj_points[i - 1][0]
                vy = traj_points[i][1] - traj_points[i - 1][1]
                speed = np.sqrt(vx ** 2 + vy ** 2)
                direction = np.arctan2(vy, vx) if speed > 0 else 0

            # 计算加速度
            if i <= 1:
                ax, ay = 0, 0
                accel = 0
            else:
                prev_vx = traj_points[i - 1][0] - traj_points[i - 2][0]
                prev_vy = traj_points[i - 1][1] - traj_points[i - 2][1]
                ax = vx - prev_vx
                ay = vy - prev_vy
                accel = np.sqrt(ax ** 2 + ay ** 2)

            # 计算jerk（加速度的变化率）
            if i <= 2:
                jerk = 0
            else:
                prev_ax = (traj_points[i - 1][0] - traj_points[i - 2][0]) - (
                            traj_points[i - 2][0] - traj_points[i - 3][0])
                prev_ay = (traj_points[i - 1][1] - traj_points[i - 2][1]) - (
                            traj_points[i - 2][1] - traj_points[i - 3][1])
                jerk_x = ax - prev_ax
                jerk_y = ay - prev_ay
                jerk = np.sqrt(jerk_x ** 2 + jerk_y ** 2)

            # 边界框变化率
            if i == 0:
                width_change = 0
                height_change = 0
            else:
                width_change = width - traj_points[i - 1][2]
                height_change = height - traj_points[i - 1][3]

            # 计算曲率（方向变化率）
            if i <= 1:
                curvature = 0
            else:
                prev_direction = np.arctan2(
                    traj_points[i - 1][1] - traj_points[i - 2][1],
                    traj_points[i - 1][0] - traj_points[i - 2][0]
                ) if i > 1 else 0
                curvature = direction - prev_direction

            # 组合特征
            features = np.array([
                center_x, center_y,  # 位置
                width, height,  # 尺寸
                vx, vy,  # 速度分量
                speed,  # 速度大小
                direction,  # 运动方向
                ax, ay,  # 加速度分量
                accel,  # 加速度大小
                jerk,  # jerk
                width_change, height_change,  # 尺寸变化
                curvature,  # 曲率
                width / height if height > 0 else 1,  # 宽高比
                np.log(speed + 1e-6)  # 对数速度
            ], dtype=np.float32)

            traj_with_features.append(features)

        return traj_with_features

    def _augment_perturb(self, trajectory):
        """数据增强：添加轻微扰动"""
        perturbed = []
        for point in trajectory:
            noise = np.random.normal(0, 0.01, point.shape)
            perturbed.append(point + noise)
        return perturbed

    def _augment_temporal_scale(self, trajectory):
        """数据增强：时间缩放"""
        if len(trajectory) <= 10:
            return trajectory

        scale = random.uniform(0.8, 1.2)
        new_length = max(5, int(len(trajectory) * scale))

        if new_length > len(trajectory):
            # 上采样：插值
            indices = np.linspace(0, len(trajectory) - 1, new_length)
            scaled = []
            for idx in indices:
                idx_floor = int(np.floor(idx))
                idx_ceil = min(int(np.ceil(idx)), len(trajectory) - 1)
                weight = idx - idx_floor

                if idx_floor == idx_ceil:
                    scaled.append(trajectory[idx_floor])
                else:
                    interpolated = (1 - weight) * trajectory[idx_floor] + weight * trajectory[idx_ceil]
                    scaled.append(interpolated)
            return scaled
        else:
            # 下采样：均匀采样
            indices = np.linspace(0, len(trajectory) - 1, new_length, dtype=int)
            return [trajectory[i] for i in indices]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.tensor(self.labels[idx], dtype=torch.long),
            torch.tensor(self.sequence_lengths[idx], dtype=torch.long)
        )


class EnhancedLSTMModel(nn.Module):
    """增强的LSTM车辆行为识别模型"""

    def __init__(self, input_dim=17, hidden_dim=256, num_layers=3, num_classes=3, dropout=0.4):
        super(EnhancedLSTMModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            proj_size=0
        )

        # 多层注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 时间特征提取器
        self.time_conv = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim // 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
                # 设置遗忘门偏置为1
                if len(param.shape) > 1:
                    n = param.size(0)
                    param.data[n // 4:n // 2].fill_(1.0)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, lengths):
        batch_size = x.size(0)
        seq_len = x.size(1)

        # 打包序列以处理变长序列
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # LSTM前向传播
        packed_output, (hidden, cell) = self.lstm(packed_input)

        # 解包序列
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # 注意力机制
        attention_weights = torch.softmax(self.attention(output).squeeze(-1), dim=1)
        attention_weights = attention_weights.unsqueeze(-1)
        context_vector = torch.sum(attention_weights * output, dim=1)

        # 时间卷积特征提取
        output_transposed = output.transpose(1, 2)
        time_features = self.time_conv(output_transposed)
        pooled_time_features = nn.functional.adaptive_avg_pool1d(time_features, 1).squeeze(-1)

        # 组合特征
        combined_features = torch.cat([context_vector, pooled_time_features], dim=1)

        # 分类
        logits = self.classifier(combined_features)

        return logits, attention_weights.squeeze(-1)


def train_epoch(model, dataloader, criterion, optimizer, device, grad_clip=1.0):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch_idx, (sequences, labels, lengths) in enumerate(pbar):
        sequences = sequences.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()

        # 前向传播
        outputs, _ = model(sequences, lengths)
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * (predicted == labels).sum().item() / labels.size(0):.2f}%'
        })

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    return avg_loss, accuracy, f1


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []
    all_confidences = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", leave=False)
        for sequences, labels, lengths in pbar:
            sequences = sequences.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            outputs, _ = model(sequences, lengths)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            probabilities = torch.softmax(outputs, dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(probabilities.max(dim=1)[0].cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    return avg_loss, accuracy, f1, all_predictions, all_labels, all_confidences


def plot_training_history(train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s):
    """绘制训练历史"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    # 损失曲线
    axes[0].plot(train_losses, label='Train Loss', linewidth=2)
    axes[0].plot(val_losses, label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')

    # 准确率曲线
    axes[1].plot(train_accs, label='Train Accuracy', linewidth=2)
    axes[1].plot(val_accs, label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 105)

    # F1分数曲线
    axes[2].plot(train_f1s, label='Train F1 Score', linewidth=2)
    axes[2].plot(val_f1s, label='Validation F1 Score', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1 Score')
    axes[2].set_title('Training and Validation F1 Score')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('training_history_improved.png', dpi=150, bbox_inches='tight')
    print("Training history saved as 'training_history_improved.png'")
    plt.close()


def plot_confusion_matrix(cm, class_names):
    """绘制混淆矩阵"""
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # 设置刻度标签
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')

    # 旋转x轴标签
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 在格子中显示数值
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("Confusion matrix saved as 'confusion_matrix.png'")
    plt.close()


def main():
    # 配置参数
    config = {
        'data_dir': '/home/next_lb/桌面/next/CAR_DETECTION_TRACK/data/BiLSTM_Transformer_data/train_data/',
        'label_dir': '/home/next_lb/桌面/next/CAR_DETECTION_TRACK/data/BiLSTM_Transformer_data/train_label/',
        'max_seq_len': 100,
        'batch_size': 64,
        'hidden_dim': 256,
        'num_layers': 3,
        'dropout': 0.4,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'num_epochs': 100,
        'patience': 15,
        'min_lr': 1e-6,
        'grad_clip': 1.0,
        'augment': True
    }

    print("=" * 60)
    print("ENHANCED VEHICLE BEHAVIOR LSTM TRAINING")
    print("=" * 60)

    # 创建数据集
    print("\nLoading dataset...")
    dataset = VehicleTrajectoryDataset(
        data_dir=config['data_dir'],
        label_dir=config['label_dir'],
        max_seq_len=config['max_seq_len'],
        augment=config['augment']
    )

    # 数据集统计
    print(f"\nDataset statistics:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Sequence shape: {dataset.sequences[0].shape}")
    print(f"  Features per timestep: {dataset.sequences[0].shape[1]}")

    class_counts = Counter(dataset.labels)
    for i, class_name in enumerate(['Normal', 'Rash', 'Accident']):
        print(f"  {class_name}: {class_counts.get(i, 0)} samples")

    # 分割数据集（80%训练，10%验证，10%测试）
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"\nDataset split:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True
    )

    # 创建模型
    input_dim = dataset.sequences[0].shape[1]

    print(f"\nCreating enhanced LSTM model...")
    print(f"  Input dimension: {input_dim}")
    print(f"  Hidden dimension: {config['hidden_dim']}")
    print(f"  Number of layers: {config['num_layers']}")

    model = EnhancedLSTMModel(
        input_dim=input_dim,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_classes=3,
        dropout=config['dropout']
    )

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel architecture:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # 定义损失函数和优化器
    # 计算类别权重以处理不平衡数据
    class_counts = Counter(dataset.labels)
    total_samples = sum(class_counts.values())
    class_weights = torch.tensor([
        total_samples / (len(class_counts) * class_counts[i]) if i in class_counts else 1.0
        for i in range(3)
    ], dtype=torch.float32).to(device)

    print(f"\nClass weights: {class_weights.cpu().numpy()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999)
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=config['patience'],
         min_lr=config['min_lr']
    )

    # 添加warmup调度器
    warmup_epochs = 5

    def warmup_scheduler(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            return 1.0

    # 训练循环
    print(f"\nStarting training for {config['num_epochs']} epochs...")
    print("=" * 60)

    train_losses = []
    train_accuracies = []
    train_f1s = []
    val_losses = []
    val_accuracies = []
    val_f1s = []

    best_val_f1 = 0
    best_epoch = 0
    best_model_state = None
    patience_counter = 0

    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")

        # Warmup学习率调整
        if epoch < warmup_epochs:
            for param_group in optimizer.param_groups:
                param_group['lr'] = config['learning_rate'] * warmup_scheduler(epoch)

        # 训练阶段
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device, config['grad_clip']
        )

        # 验证阶段
        val_loss, val_acc, val_f1, _, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        # 更新学习率
        scheduler.step(val_loss)

        # 保存记录
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        train_f1s.append(train_f1)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_f1s.append(val_f1)

        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, F1: {train_f1:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, F1: {val_f1:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # 保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_model_state = model.state_dict().copy()
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
                'val_accuracy': val_acc,
                'config': config,
                'input_dim': input_dim,
                'class_names': ['Normal', 'Rash', 'Accident'],
                'feature_scaler': dataset.feature_scaler
            }, 'best_vehicle_behavior_model_improved.pth')

            print(f"  ✓ New best model saved with validation F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{config['patience'] + 5}")

        # 早停检查
        if patience_counter >= config['patience'] + 5:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break

        # 每10个epoch保存一次中间模型
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'config': config
            }, f'checkpoint_epoch_{epoch + 1}.pth')

    # 绘制训练历史
    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies, train_f1s, val_f1s)

    # 加载最佳模型进行测试
    print(f"\nLoading best model from epoch {best_epoch + 1} for testing...")
    model.load_state_dict(best_model_state)

    # 在测试集上评估
    test_loss, test_acc, test_f1, test_predictions, test_labels, test_confidences = evaluate(
        model, test_loader, criterion, device
    )

    print(f"\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.2f}%")
    print(f"  F1 Score: {test_f1:.4f}")

    # 计算每个类别的准确率
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1_scores, support = precision_recall_fscore_support(
        test_labels, test_predictions, labels=[0, 1, 2]
    )

    print(f"\nClass-wise performance:")
    for i, class_name in enumerate(['Normal', 'Rash', 'Accident']):
        print(
            f"  {class_name}: Precision={precision[i]:.3f}, Recall={recall[i]:.3f}, F1={f1_scores[i]:.3f}, Support={support[i]}")

    # 分类报告
    print(f"\nDetailed Classification Report:")
    print(classification_report(
        test_labels, test_predictions,
        target_names=['Normal', 'Rash', 'Accident'],
        digits=3
    ))

    # 混淆矩阵
    cm = confusion_matrix(test_labels, test_predictions)
    print(f"Confusion Matrix:")
    print(cm)
    plot_confusion_matrix(cm, ['Normal', 'Rash', 'Accident'])

    # 保存最终模型
    final_model_path = 'vehicle_behavior_final_model_improved.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'input_dim': input_dim,
        'class_names': ['Normal', 'Rash', 'Accident'],
        'feature_scaler': dataset.feature_scaler,
        'test_metrics': {
            'accuracy': test_acc,
            'f1_score': test_f1,
            'loss': test_loss
        }
    }, final_model_path)

    print(f"\nFinal model saved as '{final_model_path}'")

    # 打印模型使用说明
    print(f"\n" + "=" * 60)
    print("MODEL USAGE INSTRUCTIONS")
    print("=" * 60)
    print("1. Use this model with 'inference_LSTM_improved.py' for inference")
    print("2. Use with 'V1_demo_improved.py' for real-time behavior prediction")
    print("3. Model expects {input_dim}-dimensional features")
    print("4. Features should be normalized using the saved scaler")

    # 示例推理
    print(f"\nExample prediction on a test sample:")
    if len(test_dataset) > 0:
        sample_sequence, sample_label, sample_length = test_dataset[0]
        sample_sequence = sample_sequence.unsqueeze(0).to(device)
        sample_length = sample_length.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs, attention_weights = model(sample_sequence, sample_length)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()

        print(f"  Predicted: {['Normal', 'Rash', 'Accident'][predicted_class]} "
              f"(confidence: {probabilities[0][predicted_class]:.2%})")
        print(f"  Actual: {['Normal', 'Rash', 'Accident'][sample_label]}")


if __name__ == "__main__":
    main()




