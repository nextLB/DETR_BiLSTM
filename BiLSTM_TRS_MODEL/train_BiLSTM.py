import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from tqdm import tqdm
import random

# 修复PyTorch 2.6+的模型加载问题
import torch.serialization

torch.serialization.add_safe_globals([np._core.multiarray._reconstruct])


# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


# 1. Focal Loss（专门处理类别不平衡）
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# 2. 数据预处理和加载类（添加数据增强）
class VehicleTrajectoryDataset(Dataset):
    def __init__(self, data_path, label_path, max_seq_len=100, augment_minority=True):
        self.data_path = data_path
        self.label_path = label_path
        self.max_seq_len = max_seq_len
        self.augment_minority = augment_minority

        # 加载标签
        self.labels = self._load_labels()

        # 加载轨迹数据
        self.trajectories, self.labels_list, self.seq_lengths = self._load_trajectories()

        # 数据标准化
        self.scaler = StandardScaler()
        self._normalize_data()

        print(f"  Data loaded: {len(self.trajectories)} samples")

    def _load_labels(self):
        labels_dict = {}

        if os.path.exists(self.label_path):
            with open(self.label_path, 'r') as f:
                lines = f.readlines()

            for line in lines[2:]:
                if line.strip():
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        try:
                            vehicle_id = int(parts[0])
                            label_str = parts[1].strip()

                            if label_str == 'rash':
                                label = 1
                            elif label_str == 'accident':
                                label = 2
                            else:
                                label = 0

                            labels_dict[vehicle_id] = label
                        except ValueError:
                            continue

        return labels_dict

    def _augment_trajectory(self, trajectory_data, num_augmentations=3):
        """数据增强：对少数类轨迹进行增强"""
        augmented = []

        for _ in range(num_augmentations):
            # 添加随机噪声
            noise = np.random.normal(0, 0.01, trajectory_data.shape)
            aug_traj = trajectory_data + noise

            # 随机时间扰动
            if len(trajectory_data) > 10:
                shift = np.random.randint(-2, 3)
                if shift > 0:
                    aug_traj = np.concatenate([np.zeros((shift, 7)), aug_traj[:-shift]])
                elif shift < 0:
                    aug_traj = np.concatenate([aug_traj[-shift:], np.zeros((-shift, 7))])

            augmented.append(aug_traj)

        return augmented

    def _load_trajectories(self):
        trajectories = []
        labels_list = []
        seq_lengths = []

        if not os.path.exists(self.data_path):
            print(f"  Warning: Data file {self.data_path} not found")
            return trajectories, labels_list, seq_lengths

        with open(self.data_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"  Error loading JSON: {e}")
                return trajectories, labels_list, seq_lengths

        for vehicle_id_str, trajectory in data.items():
            if not trajectory or len(trajectory) < 2:
                continue

            try:
                vehicle_id = int(vehicle_id_str)
            except ValueError:
                continue

            label = self.labels.get(vehicle_id, 0)

            # 处理轨迹数据
            trajectory_data = []
            start_idx = 0
            first_elem = trajectory[0]

            if isinstance(first_elem, list):
                if len(first_elem) == 5 and all(x == -1 for x in first_elem):
                    start_idx = 1
            elif isinstance(first_elem, (int, float)) and first_elem == -1:
                start_idx = 1

            valid_points = 0
            for i in range(start_idx, len(trajectory)):
                point = trajectory[i]

                if isinstance(point, list) and len(point) >= 4:
                    try:
                        x1 = float(point[0]) if point[0] is not None else 0.0
                        y1 = float(point[1]) if point[1] is not None else 0.0
                        x2 = float(point[2]) if point[2] is not None else 0.0
                        y2 = float(point[3]) if point[3] is not None else 0.0

                        # 计算特征
                        x_center = (x1 + x2) / 2.0
                        y_center = (y1 + y2) / 2.0
                        width = abs(x2 - x1)
                        height = abs(y2 - y1)

                        # 归一化
                        x_center_norm = x_center / 1280.0
                        y_center_norm = y_center / 720.0
                        width_norm = width / 1280.0
                        height_norm = height / 720.0

                        # 计算速度
                        if len(trajectory_data) > 0:
                            prev_x = trajectory_data[-1][0]
                            prev_y = trajectory_data[-1][1]
                            velocity_x = x_center_norm - prev_x
                            velocity_y = y_center_norm - prev_y
                        else:
                            velocity_x = 0.0
                            velocity_y = 0.0

                        # 帧号归一化
                        frame = float(point[4]) if len(point) > 4 and point[4] is not None else i
                        frame_norm = frame / 1000.0

                        trajectory_data.append([
                            x_center_norm, y_center_norm,
                            width_norm, height_norm,
                            velocity_x, velocity_y,
                            frame_norm
                        ])
                        valid_points += 1

                    except (ValueError, TypeError, IndexError):
                        continue

            if valid_points < 5:
                continue

            # 限制最大长度
            if valid_points > self.max_seq_len:
                trajectory_data = trajectory_data[:self.max_seq_len]
                seq_length = self.max_seq_len
            else:
                seq_length = valid_points

            # 填充到最大长度
            padded_trajectory = np.zeros((self.max_seq_len, 7))
            padded_trajectory[:seq_length] = trajectory_data

            # 添加原始样本
            trajectories.append(padded_trajectory)
            labels_list.append(label)
            seq_lengths.append(seq_length)

            # 对少数类进行数据增强
            if self.augment_minority and label in [1, 2] and valid_points >= 10:
                # 每个少数类样本生成3个增强样本
                for _ in range(3):
                    augmented = self._augment_trajectory(padded_trajectory[:seq_length])
                    if len(augmented) > 0:
                        aug_traj = augmented[0]
                        aug_length = min(len(aug_traj), self.max_seq_len)

                        aug_padded = np.zeros((self.max_seq_len, 7))
                        aug_padded[:aug_length] = aug_traj[:aug_length]

                        trajectories.append(aug_padded)
                        labels_list.append(label)
                        seq_lengths.append(aug_length)

        return np.array(trajectories), np.array(labels_list), np.array(seq_lengths)

    def _normalize_data(self):
        if len(self.trajectories) > 0:
            original_shape = self.trajectories.shape
            flattened = self.trajectories.reshape(-1, 7)

            non_zero_mask = np.any(flattened != 0, axis=1)
            if non_zero_mask.sum() > 0:
                self.scaler.fit(flattened[non_zero_mask])
                flattened[non_zero_mask] = self.scaler.transform(flattened[non_zero_mask])

            self.trajectories = flattened.reshape(original_shape)

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        trajectory = torch.FloatTensor(self.trajectories[idx])
        label = torch.LongTensor([self.labels_list[idx]]).squeeze()
        seq_len = torch.LongTensor([self.seq_lengths[idx]]).squeeze()

        return trajectory, label, seq_len


# 3. BiLSTM+Transformer模型（保持不变）
class BiLSTMTransformer(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=128, num_layers=2,
                 num_heads=8, dropout=0.2, num_classes=3):
        super(BiLSTMTransformer, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.lstm_fc = nn.Linear(hidden_dim * 2, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        self.ln1 = nn.LayerNorm(hidden_dim * 2)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x, seq_lens=None):
        batch_size, seq_len, _ = x.shape

        lstm_out, _ = self.lstm(x)
        lstm_out = self.ln1(lstm_out)
        lstm_out = self.dropout(lstm_out)

        lstm_out = self.lstm_fc(lstm_out)
        lstm_out = self.ln2(lstm_out)

        if seq_lens is not None:
            mask = self._create_mask(seq_lens, seq_len).to(x.device)
        else:
            mask = None

        transformer_out = self.transformer_encoder(lstm_out, src_key_padding_mask=mask)

        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(transformer_out)
            transformer_out = transformer_out.masked_fill(mask_expanded, 0)
            valid_counts = (~mask).float().sum(dim=1, keepdim=True)
            pooled = transformer_out.sum(dim=1) / valid_counts.clamp(min=1)
        else:
            pooled = transformer_out.mean(dim=1)

        output = self.fc(pooled)

        return output

    def _create_mask(self, seq_lens, max_len):
        batch_size = len(seq_lens)
        mask = torch.ones(batch_size, max_len, dtype=torch.bool)
        for i, length in enumerate(seq_lens):
            if length < max_len:
                mask[i, length:] = False
        return mask


# 4. 训练和验证函数
def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} Training')
    for trajectories, labels, seq_lens in pbar:
        trajectories = trajectories.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(trajectories, seq_lens)
        loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })

    return total_loss / len(dataloader), 100. * correct / total


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    class_correct = [0, 0, 0]
    class_total = [0, 0, 0]

    with torch.no_grad():
        for trajectories, labels, seq_lens in tqdm(dataloader, desc='Validating'):
            trajectories = trajectories.to(device)
            labels = labels.to(device)

            outputs = model(trajectories, seq_lens)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for i in range(len(labels)):
                label = labels[i].item()
                if label < 3:
                    class_total[label] += 1
                    if predicted[i] == labels[i]:
                        class_correct[label] += 1

    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total

    print(f"  Class-wise accuracy:")
    for i, class_name in enumerate(['Normal', 'Rash', 'Accident']):
        if class_total[i] > 0:
            class_acc = 100. * class_correct[i] / class_total[i]
            print(f"    Class {i} ({class_name}): {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})")

    return avg_loss, accuracy


# 5. 主训练流程
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # 数据路径
    train_data_dir = "/home/next_lb/桌面/next/CAR_DETECTION_TRACK/data/BiLSTM_Transformer_data/train_data/"
    train_label_dir = "/home/next_lb/桌面/next/CAR_DETECTION_TRACK/data/BiLSTM_Transformer_data/train_label/"

    # 加载数据
    data_files = [f for f in os.listdir(train_data_dir) if f.endswith('_extract_data.json')]
    print(f"Found {len(data_files)} data files")

    all_trajectories, all_labels, all_seq_lengths = [], [], []

    for data_file in data_files:
        label_file = data_file.replace('_extract_data.json', '_label.txt')
        data_path = os.path.join(train_data_dir, data_file)
        label_path = os.path.join(train_label_dir, label_file)

        if not os.path.exists(label_path):
            continue

        print(f"Loading {data_file}...")
        try:
            dataset = VehicleTrajectoryDataset(
                data_path=data_path,
                label_path=label_path,
                max_seq_len=100,
                augment_minority=True  # 启用数据增强
            )

            if len(dataset) > 0:
                all_trajectories.append(dataset.trajectories)
                all_labels.append(dataset.labels_list)
                all_seq_lengths.append(dataset.seq_lengths)
                print(f"  Loaded {len(dataset)} samples")
        except Exception as e:
            print(f"  Error loading {data_file}: {e}")
            continue

    if len(all_trajectories) == 0:
        print("No data loaded.")
        return

    # 合并数据
    all_trajectories = np.concatenate(all_trajectories, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_seq_lengths = np.concatenate(all_seq_lengths, axis=0)

    print(f"\nTotal samples: {len(all_trajectories)}")
    print(
        f"Class distribution: Normal: {(all_labels == 0).sum()}, Rash: {(all_labels == 1).sum()}, Accident: {(all_labels == 2).sum()}")

    # 创建数据集
    class FullDataset(Dataset):
        def __init__(self, trajectories, labels, seq_lengths):
            self.trajectories = torch.FloatTensor(trajectories)
            self.labels = torch.LongTensor(labels)
            self.seq_lengths = torch.LongTensor(seq_lengths)

        def __len__(self):
            return len(self.trajectories)

        def __getitem__(self, idx):
            return self.trajectories[idx], self.labels[idx], self.seq_lengths[idx]

    full_dataset = FullDataset(all_trajectories, all_labels, all_seq_lengths)

    # 划分训练验证集
    train_indices, val_indices = train_test_split(
        np.arange(len(full_dataset)),
        test_size=0.2,
        stratify=all_labels,
        random_state=42
    )

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # 数据加载器
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 创建模型
    model = BiLSTMTransformer(
        input_dim=7, hidden_dim=128, num_layers=2,
        num_heads=8, dropout=0.2, num_classes=3
    ).to(device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 使用Focal Loss处理类别不平衡
    class_counts = np.bincount(all_labels)
    print(f"\nClass counts: {class_counts}")

    # 计算类别权重（给少数类更大权重）
    class_weights = torch.FloatTensor([0.1, 0.4, 0.5]).to(device)  # 手动调整权重

    # 使用Focal Loss
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    # 训练参数
    num_epochs = 50
    best_val_acc = 0
    patience_counter = 0
    patience_limit = 10

    print("\nStarting training with Focal Loss and data augmentation...")

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch}/{num_epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        print(f"\n  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # 保存最佳模型（修复加载问题）
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'class_weights': class_weights.cpu().numpy(),
            }, 'best_model.pth')
            print(f"  ✓ Saved best model with val acc: {val_acc:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1

        # if patience_counter >= patience_limit:
        #     print(f"\nEarly stopping triggered")
        #     break

    print(f"\nTraining completed. Best validation accuracy: {best_val_acc:.2f}%")

    # 安全加载模型进行评估
    if os.path.exists('best_model.pth'):
        try:
            checkpoint = torch.load('best_model.pth', weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("\nLoaded best model for final evaluation")
        except Exception as e:
            print(f"\nCould not load best model: {e}")

    # 最终评估
    print("\nFinal evaluation:")
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    # 保存配置
    model_config = {
        'input_dim': 7, 'hidden_dim': 128, 'num_layers': 2,
        'num_heads': 8, 'dropout': 0.2, 'num_classes': 3,
        'best_val_acc': best_val_acc
    }

    with open('model_config.json', 'w') as f:
        json.dump(model_config, f, indent=4)

    print(f"\nModel saved to best_model.pth")
    print(f"Config saved to model_config.json")


if __name__ == "__main__":
    main()