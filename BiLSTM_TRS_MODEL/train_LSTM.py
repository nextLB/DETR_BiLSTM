import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class VehicleTrajectoryDataset(Dataset):
    """车辆轨迹数据集类"""

    def __init__(self, data_dir, label_dir, max_seq_len=100):
        """
        初始化数据集

        Args:
            data_dir: 数据文件目录
            label_dir: 标签文件目录
            max_seq_len: 最大序列长度（填充/截断）
        """
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.max_seq_len = max_seq_len
        self.sequences = []
        self.labels = []
        self.sequence_lengths = []

        # 获取所有数据文件
        data_files = [f for f in os.listdir(data_dir) if f.endswith('_extract_data.json')]

        for data_file in data_files:
            # 构建对应的标签文件名
            base_name = data_file.replace('_extract_data.json', '')
            label_file = f"{base_name}_label.txt"

            if not os.path.exists(os.path.join(label_dir, label_file)):
                print(f"Warning: Label file {label_file} not found for {data_file}")
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
            # 第一行: fps,30
            # 第二行: 1280,720
            # 后续行: id,label
            label_dict = {}
            for line in label_lines[2:]:
                line = line.strip()
                if line:
                    parts = line.split(',')
                    if len(parts) >= 2:
                        vehicle_id = parts[0]
                        label_str = parts[1].lower()
                        # 将标签映射为数字
                        if label_str == 'rash':
                            label = 1
                        elif label_str == 'accident':
                            label = 2
                        else:
                            label = 0  # 其他情况设为正常
                        label_dict[vehicle_id] = label

            # 处理每个车辆的轨迹
            for vehicle_id, trajectory in data.items():
                # 跳过第一个[-1,-1,-1,-1,-1]
                if len(trajectory) < 2:
                    continue

                # 提取轨迹点（跳过第一个-1数组）
                traj_points = []
                for point in trajectory[1:]:
                    if isinstance(point, list) and len(point) >= 4:
                        # 只取前4个值：x1, y1, x2, y2
                        x1, y1, x2, y2 = point[:4]
                        # 计算边界框中心点、宽度、高度
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        width = x2 - x1
                        height = y2 - y1
                        # 添加速度和加速度特征（使用相邻帧计算）
                        traj_points.append([center_x, center_y, width, height])

                if len(traj_points) < 3:  # 太短的序列跳过
                    continue

                # 计算速度特征
                traj_with_features = self._add_motion_features(traj_points)

                # 截断或填充序列
                if len(traj_with_features) > self.max_seq_len:
                    traj_with_features = traj_with_features[:self.max_seq_len]
                else:
                    # 填充
                    padding = [[0, 0, 0, 0, 0, 0]] * (self.max_seq_len - len(traj_with_features))
                    traj_with_features.extend(padding)

                # 获取标签（默认为0 - 正常）
                label = label_dict.get(vehicle_id, 0)

                self.sequences.append(traj_with_features)
                self.labels.append(label)
                self.sequence_lengths.append(min(len(traj_points), self.max_seq_len))

        # 转换为numpy数组
        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        self.sequence_lengths = np.array(self.sequence_lengths, dtype=np.int64)

        print(f"Dataset loaded: {len(self.sequences)} sequences")
        print(f"Class distribution: {Counter(self.labels)}")

    def _add_motion_features(self, traj_points):
        """添加运动特征（速度和加速度）"""
        traj_with_features = []

        for i in range(len(traj_points)):
            if i == 0:
                # 第一帧，速度为0
                vx, vy = 0, 0
                ax, ay = 0, 0
            else:
                # 计算速度（位置变化）
                vx = traj_points[i][0] - traj_points[i - 1][0]
                vy = traj_points[i][1] - traj_points[i - 1][1]

                if i == 1:
                    # 第二帧，加速度为0
                    ax, ay = 0, 0
                else:
                    # 计算加速度（速度变化）
                    prev_vx = traj_points[i - 1][0] - traj_points[i - 2][0]
                    prev_vy = traj_points[i - 1][1] - traj_points[i - 2][1]
                    ax = vx - prev_vx
                    ay = vy - prev_vy

            # 组合特征：[中心x, 中心y, 宽度, 高度, 速度x, 速度y]
            features = [
                traj_points[i][0],  # center_x
                traj_points[i][1],  # center_y
                traj_points[i][2],  # width
                traj_points[i][3],  # height
                vx,  # velocity_x
                vy  # velocity_y
            ]
            traj_with_features.append(features)

        return traj_with_features

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.tensor(self.labels[idx], dtype=torch.long),
            torch.tensor(self.sequence_lengths[idx], dtype=torch.long)
        )


class LSTMModel(nn.Module):
    """LSTM车辆行为识别模型"""

    def __init__(self, input_dim=6, hidden_dim=128, num_layers=2, num_classes=3, dropout=0.3):
        super(LSTMModel, self).__init__()

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
            bidirectional=True  # 使用双向LSTM捕获前后信息
        )

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
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
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, lengths):
        batch_size = x.size(0)

        # 打包序列以处理变长序列
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # LSTM前向传播
        packed_output, (hidden, cell) = self.lstm(packed_input)

        # 解包序列
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # 注意力机制
        attention_weights = torch.softmax(self.attention(output), dim=1)
        context_vector = torch.sum(attention_weights * output, dim=1)

        # 分类
        logits = self.classifier(context_vector)

        return logits


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (sequences, labels, lengths) in enumerate(dataloader):
        sequences = sequences.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()

        # 前向传播
        outputs = model(sequences, lengths)
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if batch_idx % 50 == 0:
            print(f'  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}')

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for sequences, labels, lengths in dataloader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            outputs = model(sequences, lengths)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy, all_predictions, all_labels


def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 损失曲线
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # 准确率曲线
    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history saved as 'training_history.png'")
    plt.close()


def main():
    # 配置参数
    config = {
        'data_dir': '/home/next_lb/桌面/next/CAR_DETECTION_TRACK/data/BiLSTM_Transformer_data/train_data/',
        'label_dir': '/home/next_lb/桌面/next/CAR_DETECTION_TRACK/data/BiLSTM_Transformer_data/train_label/',
        'max_seq_len': 100,  # 最大序列长度
        'batch_size': 32,
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.3,
        'learning_rate': 0.001,
        'num_epochs': 50,
        'model_type': 'lstm'  # 'lstm' 或 'gru'
    }

    # 创建数据集
    print("Loading dataset...")
    dataset = VehicleTrajectoryDataset(
        data_dir=config['data_dir'],
        label_dir=config['label_dir'],
        max_seq_len=config['max_seq_len']
    )

    # 数据集统计
    print(f"Total samples: {len(dataset)}")
    print(f"Sequence shape: {dataset.sequences[0].shape}")
    print(f"Features per timestep: {dataset.sequences[0].shape[1]}")

    # 分割数据集（80%训练，10%验证，10%测试）
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    print(f"\nDataset split:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2
    )

    # 创建模型
    input_dim = dataset.sequences[0].shape[1]  # 特征维度

    if config['model_type'] == 'lstm':
        model = LSTMModel(
            input_dim=input_dim,
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_classes=3,
            dropout=config['dropout']
        )
    else:
        # 如果需要GRU模型，这里可以添加
        pass

    model = model.to(device)
    print(f"\nModel architecture ({config['model_type'].upper()}):")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    # 定义损失函数和优化器
    # 计算类别权重以处理不平衡数据
    class_counts = Counter(dataset.labels)
    total_samples = sum(class_counts.values())
    class_weights = torch.tensor([
        total_samples / (len(class_counts) * class_counts[i]) if i in class_counts else 1.0
        for i in range(3)
    ], dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # 修改：去掉verbose参数
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # 训练循环
    print(f"\nStarting training for {config['num_epochs']} epochs...")

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    best_val_accuracy = 0
    best_model_state = None

    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")

        # 训练阶段
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # 验证阶段
        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        # 更新学习率
        scheduler.step(val_loss)

        # 保存记录
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # 保存最佳模型
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_model_state = model.state_dict().copy()
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': best_val_accuracy,
                'config': config
            }, 'best_vehicle_behavior_model.pth')
            print(f"  New best model saved with validation accuracy: {best_val_accuracy:.2f}%")

    # 绘制训练历史
    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)

    # 加载最佳模型进行测试
    print("\nLoading best model for testing...")
    model.load_state_dict(best_model_state)

    # 在测试集上评估
    test_loss, test_acc, test_predictions, test_labels = evaluate(
        model, test_loader, criterion, device
    )

    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.2f}%")

    # 分类报告
    print("\nClassification Report:")
    print(classification_report(
        test_labels, test_predictions,
        target_names=['Normal (0)', 'Rash (1)', 'Accident (2)']
    ))

    # 混淆矩阵
    cm = confusion_matrix(test_labels, test_predictions)
    print("Confusion Matrix:")
    print(cm)

    # 保存最终模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'input_dim': input_dim,
        'class_names': ['Normal', 'Rash', 'Accident']
    }, 'vehicle_behavior_final_model.pth')

    print("\nModel saved as 'vehicle_behavior_final_model.pth'")

    # 示例推理函数
    def predict_single_trajectory(trajectory_points, model, device, max_seq_len=100):
        """预测单个轨迹的行为"""
        model.eval()

        # 预处理轨迹点
        if len(trajectory_points) < 3:
            return "轨迹太短，无法预测", 0.0

        # 添加运动特征
        traj_with_features = []
        for i in range(len(trajectory_points)):
            if i == 0:
                vx, vy = 0, 0
            else:
                vx = trajectory_points[i][0] - trajectory_points[i - 1][0]
                vy = trajectory_points[i][1] - trajectory_points[i - 1][1]

            features = [
                trajectory_points[i][0],  # center_x
                trajectory_points[i][1],  # center_y
                trajectory_points[i][2],  # width
                trajectory_points[i][3],  # height
                vx,  # velocity_x
                vy  # velocity_y
            ]
            traj_with_features.append(features)

        # 截断或填充
        if len(traj_with_features) > max_seq_len:
            traj_with_features = traj_with_features[:max_seq_len]
        else:
            padding = [[0, 0, 0, 0, 0, 0]] * (max_seq_len - len(traj_with_features))
            traj_with_features.extend(padding)

        # 转换为tensor
        sequence = torch.FloatTensor([traj_with_features]).to(device)
        length = torch.tensor([min(len(trajectory_points), max_seq_len)], dtype=torch.long).to(device)

        # 预测
        with torch.no_grad():
            outputs = model(sequence, length)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()

        class_names = ['Normal', 'Rash', 'Accident']
        confidence = probabilities[0][predicted_class].item()

        return class_names[predicted_class], confidence

    # 测试示例推理
    print("\nExample prediction:")
    if len(dataset) > 0:
        sample_sequence = dataset.sequences[0][:10]  # 取前10个时间步
        sample_points = [[seq[0], seq[1], seq[2], seq[3]] for seq in sample_sequence]

        prediction, confidence = predict_single_trajectory(
            sample_points, model, device, config['max_seq_len']
        )
        print(f"Predicted: {prediction} (confidence: {confidence:.2%})")
        print(f"Actual label: {['Normal', 'Rash', 'Accident'][dataset.labels[0]]}")


if __name__ == "__main__":
    main()


# import json
# import os
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
# from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
# from sklearn.utils.class_weight import compute_class_weight
# import matplotlib.pyplot as plt
# from collections import Counter
# import warnings
#
# warnings.filterwarnings('ignore')
#
# # 设置随机种子
# torch.manual_seed(42)
# np.random.seed(42)
#
# # 定义设备
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")
#
#
# # 定义Focal Loss来处理类别不平衡
# class FocalLoss(nn.Module):
#     def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         self.reduction = reduction
#
#     def forward(self, inputs, targets):
#         ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
#         pt = torch.exp(-ce_loss)
#         focal_loss = ((1 - pt) ** self.gamma) * ce_loss
#
#         if self.alpha is not None:
#             alpha_t = self.alpha[targets]
#             focal_loss = alpha_t * focal_loss
#
#         if self.reduction == 'mean':
#             return focal_loss.mean()
#         elif self.reduction == 'sum':
#             return focal_loss.sum()
#         else:
#             return focal_loss
#
#
# class VehicleTrajectoryDataset(Dataset):
#     """改进的车辆轨迹数据集类，添加数据增强"""
#
#     def __init__(self, data_dir, label_dir, max_seq_len=100, augment=False):
#         self.data_dir = data_dir
#         self.label_dir = label_dir
#         self.max_seq_len = max_seq_len
#         self.augment = augment
#         self.sequences = []
#         self.labels = []
#         self.sequence_lengths = []
#
#         # 获取所有数据文件
#         data_files = [f for f in os.listdir(data_dir) if f.endswith('_extract_data.json')]
#
#         for data_file in data_files:
#             base_name = data_file.replace('_extract_data.json', '')
#             label_file = f"{base_name}_label.txt"
#
#             if not os.path.exists(os.path.join(label_dir, label_file)):
#                 continue
#
#             # 加载数据
#             data_path = os.path.join(data_dir, data_file)
#             with open(data_path, 'r') as f:
#                 data = json.load(f)
#
#             # 加载标签
#             label_path = os.path.join(label_dir, label_file)
#             with open(label_path, 'r') as f:
#                 label_lines = f.readlines()
#
#             # 解析标签文件
#             label_dict = {}
#             for line in label_lines[2:]:
#                 line = line.strip()
#                 if line:
#                     parts = line.split(',')
#                     if len(parts) >= 2:
#                         vehicle_id = parts[0]
#                         label_str = parts[1].lower()
#                         if label_str == 'rash':
#                             label = 1
#                         elif label_str == 'accident':
#                             label = 2
#                         else:
#                             label = 0
#                         label_dict[vehicle_id] = label
#
#             # 处理每个车辆的轨迹
#             for vehicle_id, trajectory in data.items():
#                 if len(trajectory) < 2:
#                     continue
#
#                 # 提取轨迹点
#                 traj_points = []
#                 for point in trajectory[1:]:
#                     if isinstance(point, list) and len(point) >= 4:
#                         x1, y1, x2, y2 = point[:4]
#                         center_x = (x1 + x2) / 2
#                         center_y = (y1 + y2) / 2
#                         width = x2 - x1
#                         height = y2 - y1
#                         traj_points.append([center_x, center_y, width, height])
#
#                 if len(traj_points) < 3:
#                     continue
#
#                 # 获取标签
#                 label = label_dict.get(vehicle_id, 0)
#
#                 # 添加原始样本
#                 self._add_sequence(traj_points, label)
#
#                 # 对少数类进行数据增强
#                 if self.augment and label != 0:
#                     self._augment_sequence(traj_points, label)
#
#         # 转换为numpy数组
#         self.sequences = np.array(self.sequences, dtype=np.float32)
#         self.labels = np.array(self.labels, dtype=np.int64)
#         self.sequence_lengths = np.array(self.sequence_lengths, dtype=np.int64)
#
#         print(f"Dataset loaded: {len(self.sequences)} sequences")
#         print(f"Class distribution after augmentation: {Counter(self.labels)}")
#
#     def _add_sequence(self, traj_points, label):
#         """添加序列到数据集"""
#         traj_with_features = self._add_motion_features(traj_points)
#
#         # 截断或填充序列
#         if len(traj_with_features) > self.max_seq_len:
#             traj_with_features = traj_with_features[:self.max_seq_len]
#         else:
#             padding = [[0, 0, 0, 0, 0, 0]] * (self.max_seq_len - len(traj_with_features))
#             traj_with_features.extend(padding)
#
#         self.sequences.append(traj_with_features)
#         self.labels.append(label)
#         self.sequence_lengths.append(min(len(traj_points), self.max_seq_len))
#
#     def _augment_sequence(self, traj_points, label):
#         """对少数类进行数据增强"""
#         # 1. 添加高斯噪声
#         if np.random.random() > 0.5:
#             noise_traj = []
#             for point in traj_points:
#                 noise_point = [
#                     point[0] + np.random.normal(0, 2),  # center_x
#                     point[1] + np.random.normal(0, 2),  # center_y
#                     max(10, point[2] + np.random.normal(0, 1)),  # width
#                     max(10, point[3] + np.random.normal(0, 1)),  # height
#                 ]
#                 noise_traj.append(noise_point)
#             self._add_sequence(noise_traj, label)
#
#         # 2. 时间缩放（仅对足够长的序列）
#         if len(traj_points) > 10 and np.random.random() > 0.5:
#             scale_factor = np.random.uniform(0.8, 1.2)
#             scaled_traj = []
#             for i in range(len(traj_points)):
#                 idx = min(int(i * scale_factor), len(traj_points) - 1)
#                 scaled_traj.append(traj_points[idx])
#             self._add_sequence(scaled_traj, label)
#
#     def _add_motion_features(self, traj_points):
#         """添加运动特征"""
#         traj_with_features = []
#
#         for i in range(len(traj_points)):
#             if i == 0:
#                 vx, vy = 0, 0
#             else:
#                 vx = traj_points[i][0] - traj_points[i - 1][0]
#                 vy = traj_points[i][1] - traj_points[i - 1][1]
#
#             features = [
#                 traj_points[i][0],  # center_x
#                 traj_points[i][1],  # center_y
#                 traj_points[i][2],  # width
#                 traj_points[i][3],  # height
#                 vx,  # velocity_x
#                 vy  # velocity_y
#             ]
#             traj_with_features.append(features)
#
#         return traj_with_features
#
#     def __len__(self):
#         return len(self.sequences)
#
#     def __getitem__(self, idx):
#         return (
#             torch.FloatTensor(self.sequences[idx]),
#             torch.tensor(self.labels[idx], dtype=torch.long),
#             torch.tensor(self.sequence_lengths[idx], dtype=torch.long)
#         )
#
#
# class ImprovedLSTMModel(nn.Module):
#     """改进的LSTM模型，添加Dropout和BatchNorm"""
#
#     def __init__(self, input_dim=6, hidden_dim=128, num_layers=2, num_classes=3, dropout=0.5):
#         super(ImprovedLSTMModel, self).__init__()
#
#         self.lstm = nn.LSTM(
#             input_size=input_dim,
#             hidden_size=hidden_dim,
#             num_layers=num_layers,
#             batch_first=True,
#             dropout=dropout if num_layers > 1 else 0,
#             bidirectional=True
#         )
#
#         # 添加BatchNorm
#         self.bn = nn.BatchNorm1d(hidden_dim * 2)
#
#         self.attention = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.Tanh(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, 1)
#         )
#
#         self.classifier = nn.Sequential(
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.ReLU(),
#             nn.BatchNorm1d(hidden_dim),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.ReLU(),
#             nn.BatchNorm1d(hidden_dim // 2),
#             nn.Linear(hidden_dim // 2, num_classes)
#         )
#
#         self._init_weights()
#
#     def _init_weights(self):
#         for name, param in self.lstm.named_parameters():
#             if 'weight' in name:
#                 nn.init.orthogonal_(param)
#             elif 'bias' in name:
#                 nn.init.constant_(param, 0)
#
#     def forward(self, x, lengths):
#         packed_input = nn.utils.rnn.pack_padded_sequence(
#             x, lengths.cpu(), batch_first=True, enforce_sorted=False
#         )
#
#         packed_output, (hidden, cell) = self.lstm(packed_input)
#         output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
#
#         # 应用BatchNorm
#         output = output.permute(0, 2, 1)  # (batch, features, seq_len)
#         output = self.bn(output)
#         output = output.permute(0, 2, 1)  # (batch, seq_len, features)
#
#         attention_weights = torch.softmax(self.attention(output), dim=1)
#         context_vector = torch.sum(attention_weights * output, dim=1)
#
#         logits = self.classifier(context_vector)
#         return logits
#
#
# def train_epoch(model, dataloader, criterion, optimizer, device):
#     """训练一个epoch"""
#     model.train()
#     total_loss = 0
#     correct = 0
#     total = 0
#
#     for batch_idx, (sequences, labels, lengths) in enumerate(dataloader):
#         sequences = sequences.to(device)
#         labels = labels.to(device)
#         lengths = lengths.to(device)
#
#         optimizer.zero_grad()
#
#         outputs = model(sequences, lengths)
#         loss = criterion(outputs, labels)
#
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         optimizer.step()
#
#         total_loss += loss.item()
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
#     avg_loss = total_loss / len(dataloader)
#     accuracy = 100 * correct / total
#
#     return avg_loss, accuracy
#
#
# def evaluate(model, dataloader, criterion, device):
#     """评估模型"""
#     model.eval()
#     total_loss = 0
#     correct = 0
#     total = 0
#
#     all_predictions = []
#     all_labels = []
#     all_probabilities = []
#
#     with torch.no_grad():
#         for sequences, labels, lengths in dataloader:
#             sequences = sequences.to(device)
#             labels = labels.to(device)
#             lengths = lengths.to(device)
#
#             outputs = model(sequences, lengths)
#             loss = criterion(outputs, labels)
#
#             total_loss += loss.item()
#             probabilities = torch.softmax(outputs, dim=1)
#             _, predicted = torch.max(outputs.data, 1)
#
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#
#             all_predictions.extend(predicted.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
#             all_probabilities.extend(probabilities.cpu().numpy())
#
#     avg_loss = total_loss / len(dataloader)
#     accuracy = 100 * correct / total
#
#     return avg_loss, accuracy, all_predictions, all_labels, all_probabilities
#
#
# def main():
#     """主训练函数"""
#     # 配置参数
#     config = {
#         'data_dir': '/home/next_lb/桌面/next/CAR_DETECTION_TRACK/data/BiLSTM_Transformer_data/train_data/',
#         'label_dir': '/home/next_lb/桌面/next/CAR_DETECTION_TRACK/data/BiLSTM_Transformer_data/train_label/',
#         'max_seq_len': 50,  # 减少序列长度，防止过拟合
#         'batch_size': 16,  # 减小批量大小
#         'hidden_dim': 64,  # 减少隐藏层维度
#         'num_layers': 2,
#         'dropout': 0.5,  # 增加Dropout
#         'learning_rate': 0.0001,  # 降低学习率
#         'num_epochs': 100,
#         'patience': 10,  # 早停耐心值
#         'gamma': 2.0,  # Focal Loss参数
#     }
#
#     print("Loading and augmenting dataset...")
#     dataset = VehicleTrajectoryDataset(
#         data_dir=config['data_dir'],
#         label_dir=config['label_dir'],
#         max_seq_len=config['max_seq_len'],
#         augment=True  # 启用数据增强
#     )
#
#     # 计算类别权重
#     class_counts = Counter(dataset.labels)
#     total = len(dataset.labels)
#     class_weights = torch.tensor([
#         total / (len(class_counts) * class_counts[i]) if i in class_counts else 1.0
#         for i in range(3)
#     ], dtype=torch.float32).to(device)
#
#     print(f"Class weights: {class_weights}")
#
#     # 分割数据集
#     total_size = len(dataset)
#     train_size = int(0.7 * total_size)  # 减少训练集比例
#     val_size = int(0.15 * total_size)
#     test_size = total_size - train_size - val_size
#
#     train_dataset, val_dataset, test_dataset = random_split(
#         dataset, [train_size, val_size, test_size]
#     )
#
#     print(f"\nDataset split:")
#     print(f"  Train: {len(train_dataset)} samples")
#     print(f"  Validation: {len(val_dataset)} samples")
#     print(f"  Test: {len(test_dataset)} samples")
#
#     # 创建数据加载器
#     train_loader = DataLoader(
#         train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2
#     )
#     val_loader = DataLoader(
#         val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2
#     )
#     test_loader = DataLoader(
#         test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2
#     )
#
#     # 创建模型
#     input_dim = dataset.sequences[0].shape[1]
#     model = ImprovedLSTMModel(
#         input_dim=input_dim,
#         hidden_dim=config['hidden_dim'],
#         num_layers=config['num_layers'],
#         num_classes=3,
#         dropout=config['dropout']
#     )
#
#     model = model.to(device)
#     print(f"\nModel architecture:")
#     print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
#
#     # 定义损失函数和优化器
#     criterion = FocalLoss(alpha=class_weights, gamma=config['gamma'])
#     optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.01)
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
#
#     # 训练循环
#     print(f"\nStarting training for {config['num_epochs']} epochs...")
#
#     train_losses = []
#     train_accuracies = []
#     val_losses = []
#     val_accuracies = []
#
#     best_val_accuracy = 0
#     best_val_f1 = 0
#     patience_counter = 0
#     best_model_state = None
#
#     for epoch in range(config['num_epochs']):
#         # 训练
#         train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
#
#         # 验证
#         val_loss, val_acc, val_preds, val_labels, _ = evaluate(model, val_loader, criterion, device)
#
#         # 计算F1分数（更关注少数类）
#         _, _, f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='weighted')
#
#         # 更新学习率
#         scheduler.step()
#
#         # 保存记录
#         train_losses.append(train_loss)
#         train_accuracies.append(train_acc)
#         val_losses.append(val_loss)
#         val_accuracies.append(val_acc)
#
#         print(f"Epoch {epoch + 1}/{config['num_epochs']}: "
#               f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
#               f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {f1:.4f}")
#
#         # 保存最佳模型（基于F1分数）
#         if f1 > best_val_f1:
#             best_val_f1 = f1
#             best_val_accuracy = val_acc
#             patience_counter = 0
#             best_model_state = model.state_dict().copy()
#
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': best_model_state,
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'val_accuracy': best_val_accuracy,
#                 'val_f1': best_val_f1,
#                 'config': config,
#                 'class_weights': class_weights.cpu().numpy()
#             }, 'improved_vehicle_behavior_model.pth')
#
#             print(f"  New best model saved! F1: {best_val_f1:.4f}")
#         else:
#             patience_counter += 1
#             # if patience_counter >= config['patience']:
#             #     print(f"Early stopping triggered after {epoch + 1} epochs")
#             #     break
#
#     # 绘制训练历史
#     plt.figure(figsize=(12, 4))
#
#     plt.subplot(1, 2, 1)
#     plt.plot(train_losses, label='Train Loss')
#     plt.plot(val_losses, label='Validation Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Training and Validation Loss')
#     plt.legend()
#     plt.grid(True)
#
#     plt.subplot(1, 2, 2)
#     plt.plot(train_accuracies, label='Train Accuracy')
#     plt.plot(val_accuracies, label='Validation Accuracy')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy (%)')
#     plt.title('Training and Validation Accuracy')
#     plt.legend()
#     plt.grid(True)
#
#     plt.tight_layout()
#     plt.savefig('improved_training_history.png')
#     plt.close()
#
#     # 加载最佳模型进行测试
#     print("\nLoading best model for testing...")
#     model.load_state_dict(best_model_state)
#
#     # 在测试集上评估
#     test_loss, test_acc, test_preds, test_labels, test_probs = evaluate(
#         model, test_loader, criterion, device
#     )
#
#     print(f"\nTest Results:")
#     print(f"  Loss: {test_loss:.4f}")
#     print(f"  Accuracy: {test_acc:.2f}%")
#
#     # 分类报告
#     print("\nClassification Report:")
#     print(classification_report(test_labels, test_preds, target_names=['Normal', 'Rash', 'Accident']))
#
#     # 混淆矩阵
#     cm = confusion_matrix(test_labels, test_preds)
#     print("Confusion Matrix:")
#     print(cm)
#
#     # 调整阈值后的预测
#     print("\nAdjusted Threshold Predictions (threshold=0.3 for minority classes):")
#
#     adjusted_preds = []
#     for probs in test_probs:
#         if probs[1] > 0.3:  # Rash阈值
#             adjusted_preds.append(1)
#         elif probs[2] > 0.3:  # Accident阈值
#             adjusted_preds.append(2)
#         else:
#             adjusted_preds.append(0)
#
#     print("Adjusted Classification Report:")
#     print(classification_report(test_labels, adjusted_preds, target_names=['Normal', 'Rash', 'Accident']))
#
#     # 保存最终模型
#     torch.save({
#         'model_state_dict': model.state_dict(),
#         'config': config,
#         'input_dim': input_dim,
#         'class_names': ['Normal', 'Rash', 'Accident'],
#         'thresholds': [0.3, 0.3]  # Rash和Accident的阈值
#     }, 'improved_final_model.pth')
#
#     print("\nModel saved as 'improved_final_model.pth'")
#
#
# if __name__ == "__main__":
#     main()
