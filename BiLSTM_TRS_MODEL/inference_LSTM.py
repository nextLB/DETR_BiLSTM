
import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import warnings
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# 定义增强的数据集类
class EnhancedVehicleTrajectoryDataset(Dataset):
    """增强的车辆轨迹数据集类"""

    def __init__(self, data_dir, label_dir, max_seq_len=100, feature_scaler=None):
        """
        初始化数据集

        Args:
            data_dir: 数据文件目录
            label_dir: 标签文件目录
            max_seq_len: 最大序列长度（填充/截断）
            feature_scaler: 特征标准化器
        """
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.max_seq_len = max_seq_len
        self.feature_scaler = feature_scaler
        self.sequences = []
        self.labels = []
        self.sequence_lengths = []
        self.vehicle_ids = []
        self.raw_trajectories = []

        # 获取所有数据文件
        data_files = [f for f in os.listdir(data_dir) if f.endswith('_extract_data.json')]
        data_files.sort()

        # 用于特征标准化的数据收集
        all_features_for_scaling = []

        for data_file in tqdm(data_files, desc="Loading data"):
            # 构建对应的标签文件名
            base_name = data_file.replace('_extract_data.json', '')
            label_file = f"{base_name}_label.txt"

            # 加载数据
            data_path = os.path.join(data_dir, data_file)
            with open(data_path, 'r') as f:
                data = json.load(f)

            # 加载标签（如果存在）
            label_dict = {}
            label_path = os.path.join(label_dir, label_file)
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    label_lines = f.readlines()

                # 解析标签文件
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
            else:
                continue  # 如果没有标签文件，跳过这个数据文件

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

                # 保存原始轨迹用于调试
                self.raw_trajectories.append(traj_points)

                # 收集特征用于标准化
                all_features_for_scaling.extend(traj_with_features)

                # 获取标签
                label = label_dict.get(vehicle_id, 0)

                # 保存未标准化的特征和相关信息
                self.sequences.append({
                    'features': traj_with_features,
                    'label': label,
                    'vehicle_id': f"{data_file}_{vehicle_id}",
                    'original_length': len(traj_with_features)
                })

        print(f"\nLoaded {len(self.sequences)} sequences")

        # 如果提供了特征标准化器，使用它；否则创建一个新的
        if self.feature_scaler is None:
            print("Creating new feature scaler...")
            self.feature_scaler = StandardScaler()
            all_features_array = np.vstack(all_features_for_scaling)
            self.feature_scaler.fit(all_features_array)
            print(f"Feature scaler fitted on {all_features_array.shape[0]} samples")
        else:
            print("Using provided feature scaler")

        # 现在标准化所有特征并转换为numpy数组
        processed_sequences = []
        processed_labels = []
        processed_lengths = []
        processed_vehicle_ids = []

        for seq_info in self.sequences:
            traj_with_features = seq_info['features']
            label = seq_info['label']
            vehicle_id = seq_info['vehicle_id']
            original_length = seq_info['original_length']

            # 标准化特征
            traj_normalized = self.feature_scaler.transform(traj_with_features)

            # 截断或填充序列
            if original_length > self.max_seq_len:
                traj_normalized = traj_normalized[:self.max_seq_len]
                seq_len = self.max_seq_len
            else:
                # 创建填充（使用特征维度的零向量）
                feature_dim = traj_normalized.shape[1]
                padding = np.zeros((self.max_seq_len - original_length, feature_dim))
                traj_normalized = np.vstack([traj_normalized, padding])
                seq_len = original_length

            processed_sequences.append(traj_normalized)
            processed_labels.append(label)
            processed_lengths.append(seq_len)
            processed_vehicle_ids.append(vehicle_id)

        # 转换为numpy数组
        self.sequences = np.array(processed_sequences, dtype=np.float32)
        self.labels = np.array(processed_labels, dtype=np.int64)
        self.sequence_lengths = np.array(processed_lengths, dtype=np.int64)
        self.vehicle_ids = processed_vehicle_ids

        print(f"\nDataset statistics:")
        print(f"  Total sequences: {len(self.sequences)}")
        print(f"  Sequence shape: {self.sequences[0].shape}")
        print(f"  Feature dimension: {self.sequences[0].shape[1]}")
        print(f"  Class distribution: Normal={sum(self.labels == 0)}, "
              f"Rash={sum(self.labels == 1)}, Accident={sum(self.labels == 2)}")

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

            # 宽高比
            aspect_ratio = width / height if height > 0 else 1.0

            # 对数速度
            log_speed = np.log(speed + 1e-6)

            # 组合特征（确保是17维）
            features = np.array([
                center_x, center_y,  # 位置 (2)
                width, height,  # 尺寸 (2)
                vx, vy,  # 速度分量 (2)
                speed,  # 速度大小 (1)
                direction,  # 运动方向 (1)
                ax, ay,  # 加速度分量 (2)
                accel,  # 加速度大小 (1)
                jerk,  # jerk (1)
                width_change, height_change,  # 尺寸变化 (2)
                curvature,  # 曲率 (1)
                aspect_ratio,  # 宽高比 (1)
                log_speed  # 对数速度 (1)
            ], dtype=np.float32)

            # 确保特征维度为17
            assert len(features) == 17, f"Feature dimension is {len(features)}, expected 17"

            traj_with_features.append(features)

        return np.array(traj_with_features, dtype=np.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.tensor(self.labels[idx], dtype=torch.long),
            torch.tensor(self.sequence_lengths[idx], dtype=torch.long),
            self.vehicle_ids[idx]
        )


# 定义增强的LSTM模型
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


def load_model(model_path, device):
    """加载训练好的模型"""
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # 获取配置
    config = checkpoint['config']
    input_dim = checkpoint.get('input_dim', 17)
    feature_scaler = checkpoint.get('feature_scaler', None)
    class_names = checkpoint.get('class_names', ['Normal', 'Rash', 'Accident'])

    # 创建模型
    model = EnhancedLSTMModel(
        input_dim=input_dim,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_classes=3,
        dropout=0.0  # 推理时不需要dropout
    )

    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Model loaded successfully")
    print(
        f"Model configuration: input_dim={input_dim}, hidden_dim={config['hidden_dim']}, num_layers={config['num_layers']}")
    print(f"Class names: {class_names}")

    if feature_scaler is None:
        print("Warning: No feature scaler found in checkpoint!")

    return model, class_names, feature_scaler


def predict_dataset(model, dataloader, device, class_names):
    """对整个数据集进行预测"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_confidences = []
    all_vehicle_ids = []
    all_probabilities = []

    with torch.no_grad():
        for batch_idx, (sequences, labels, lengths, vehicle_ids) in enumerate(tqdm(dataloader, desc="Predicting")):
            sequences = sequences.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            outputs, _ = model(sequences, lengths)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probabilities, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
            all_vehicle_ids.extend(vehicle_ids)
            all_probabilities.extend(probabilities.cpu().numpy())

    return all_predictions, all_labels, all_confidences, all_vehicle_ids, all_probabilities


def print_detailed_results(predictions, labels, confidences, vehicle_ids, probabilities, class_names):
    """打印详细的预测结果"""
    print("\n" + "=" * 80)
    print("DETAILED PREDICTION RESULTS")
    print("=" * 80)

    total = len(predictions)

    # 创建结果字典
    results = []
    for i in range(total):
        pred_class = predictions[i]
        true_class = labels[i]
        confidence = confidences[i]
        vehicle_id = vehicle_ids[i]
        probs = probabilities[i]

        is_correct = (pred_class == true_class)

        result = {
            'vehicle_id': vehicle_id,
            'prediction': class_names[pred_class],
            'true_label': class_names[true_class],
            'confidence': float(confidence),
            'correct': is_correct,
            'prob_normal': float(probs[0]),
            'prob_rash': float(probs[1]),
            'prob_accident': float(probs[2])
        }
        results.append(result)

    # 计算总体指标
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')

    print(f"\nOverall Statistics:")
    print(f"  Total vehicles: {total}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"  Weighted F1 Score: {f1:.4f}")

    # 分类报告
    print(f"\nDetailed Classification Report:")
    print(classification_report(labels, predictions, target_names=class_names, digits=3))

    # 混淆矩阵
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(labels, predictions)
    print(cm)

    # 按类别统计
    print(f"\nClass-wise Performance:")
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1_scores, support = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1, 2]
    )

    for i, class_name in enumerate(class_names):
        class_indices = [j for j in range(total) if labels[j] == i]
        if class_indices:
            class_confidences = [confidences[j] for j in class_indices]
            avg_confidence = np.mean(class_confidences)
        else:
            avg_confidence = 0

        print(f"  {class_name}:")
        print(f"    Precision: {precision[i]:.3f}")
        print(f"    Recall: {recall[i]:.3f}")
        print(f"    F1-Score: {f1_scores[i]:.3f}")
        print(f"    Support: {support[i]}")
        print(f"    Average Confidence: {avg_confidence:.3f}")

    # 保存到文件
    df = pd.DataFrame(results)
    output_file = 'prediction_results_detailed.csv'
    df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to '{output_file}'")

    return results


def predict_single_trajectory(model, bbox_sequence, device, feature_scaler=None, max_seq_len=100):
    """预测单个轨迹的行为"""
    model.eval()

    if len(bbox_sequence) < 5:
        return 0, 0.0, [0.0, 0.0, 0.0], "轨迹太短，无法进行可靠预测"

    # 转换为中心点坐标和尺寸
    traj_points = []
    for bbox in bbox_sequence:
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        traj_points.append([center_x, center_y, width, height])

    # 添加丰富的运动特征
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
            prev_ax = (traj_points[i - 1][0] - traj_points[i - 2][0]) - (traj_points[i - 2][0] - traj_points[i - 3][0])
            prev_ay = (traj_points[i - 1][1] - traj_points[i - 2][1]) - (traj_points[i - 2][1] - traj_points[i - 3][1])
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

        # 宽高比
        aspect_ratio = width / height if height > 0 else 1.0

        # 对数速度
        log_speed = np.log(speed + 1e-6)

        # 组合特征（确保是17维）
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
            aspect_ratio,  # 宽高比
            log_speed  # 对数速度
        ], dtype=np.float32)

        traj_with_features.append(features)

    # 转换为numpy数组
    traj_with_features = np.array(traj_with_features, dtype=np.float32)

    # 归一化特征
    if feature_scaler is not None:
        traj_with_features = feature_scaler.transform(traj_with_features)

    # 截断或填充
    if len(traj_with_features) > max_seq_len:
        traj_with_features = traj_with_features[:max_seq_len]
        seq_len = max_seq_len
    else:
        # 创建填充（使用特征维度的零向量）
        feature_dim = traj_with_features.shape[1]
        padding = np.zeros((max_seq_len - len(traj_with_features), feature_dim))
        traj_with_features = np.vstack([traj_with_features, padding])
        seq_len = len(traj_with_features)

    # 转换为tensor
    sequence = torch.FloatTensor([traj_with_features]).to(device)
    length = torch.tensor([seq_len], dtype=torch.long).to(device)

    # 预测
    with torch.no_grad():
        outputs, attention_weights = model(sequence, length)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, dim=1).item()

    confidence = probabilities[0][predicted_class].item()
    all_probabilities = probabilities[0].cpu().numpy()

    # 生成解释文本
    if predicted_class == 0:
        explanation = "正常行驶：速度和加速度在正常范围内"
    elif predicted_class == 1:
        explanation = "危险驾驶：检测到高速、急转弯或急加速"
    else:
        explanation = "事故：检测到异常停车或碰撞模式"

    return predicted_class, confidence, all_probabilities, explanation


def main():
    """主函数"""
    print("=" * 60)
    print("ENHANCED VEHICLE BEHAVIOR LSTM INFERENCE")
    print("=" * 60)

    # 配置参数
    config = {
        'data_dir': '/home/next_lb/桌面/next/CAR_DETECTION_TRACK/data/BiLSTM_Transformer_data/train_data/',
        'label_dir': '/home/next_lb/桌面/next/CAR_DETECTION_TRACK/data/BiLSTM_Transformer_data/train_label/',
        'max_seq_len': 100,
        'batch_size': 32,
    }

    # 测试模式：'dataset' 或 'single'
    test_mode = 'dataset'  # 先测试单个轨迹

    # 加载模型
    model_paths = [
        'vehicle_behavior_final_model_improved.pth',
        'best_vehicle_behavior_model_improved.pth',
        'vehicle_behavior_final_model.pth',
        'best_vehicle_behavior_model.pth'
    ]

    model_loaded = False
    for model_path in model_paths:
        if os.path.exists(model_path):
            model, class_names, feature_scaler = load_model(model_path, device)
            model_loaded = True
            break

    if not model_loaded:
        print("Error: No model file found!")
        print("Please make sure you have trained the model first.")
        return

    if test_mode == 'dataset':
        # 测试整个数据集
        print("\nLoading dataset for prediction...")
        dataset = EnhancedVehicleTrajectoryDataset(
            data_dir=config['data_dir'],
            label_dir=config['label_dir'],
            max_seq_len=config['max_seq_len'],
            feature_scaler=feature_scaler
        )

        if len(dataset) == 0:
            print("No data found in the specified directories!")
            return

        # 创建数据加载器
        dataloader = DataLoader(
            dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2
        )

        # 进行预测
        print("\nMaking predictions on the entire dataset...")
        predictions, labels, confidences, vehicle_ids, probabilities = predict_dataset(
            model, dataloader, device, class_names
        )

        # 打印结果
        results = print_detailed_results(predictions, labels, confidences, vehicle_ids, probabilities, class_names)

    elif test_mode == 'single':
        # 测试单个轨迹
        print("\nTesting single trajectory prediction...")

        # 示例1：正常行驶轨迹
        print("\nExample 1: Normal driving")
        test_trajectory1 = []
        for i in range(30):
            x1 = 100 + i * 3  # 缓慢向右移动
            y1 = 100 + i * 1  # 缓慢向下移动
            x2 = x1 + 50
            y2 = y1 + 30
            test_trajectory1.append([x1, y1, x2, y2])

        pred_class1, confidence1, probs1, explanation1 = predict_single_trajectory(
            model, test_trajectory1, device, feature_scaler, config['max_seq_len']
        )

        print(f"  Prediction: {class_names[pred_class1]}")
        print(f"  Confidence: {confidence1:.2%}")
        print(f"  Probabilities: Normal={probs1[0]:.2%}, Rash={probs1[1]:.2%}, Accident={probs1[2]:.2%}")
        print(f"  Explanation: {explanation1}")

        # 示例2：危险驾驶轨迹
        print("\nExample 2: Rash driving")
        test_trajectory2 = []
        for i in range(30):
            x1 = 100 + i * 15  # 快速向右移动
            y1 = 100 + i * 8  # 快速向下移动
            x2 = x1 + 50
            y2 = y1 + 30
            test_trajectory2.append([x1, y1, x2, y2])

        pred_class2, confidence2, probs2, explanation2 = predict_single_trajectory(
            model, test_trajectory2, device, feature_scaler, config['max_seq_len']
        )

        print(f"  Prediction: {class_names[pred_class2]}")
        print(f"  Confidence: {confidence2:.2%}")
        print(f"  Probabilities: Normal={probs2[0]:.2%}, Rash={probs2[1]:.2%}, Accident={probs2[2]:.2%}")
        print(f"  Explanation: {explanation2}")

        # 示例3：事故轨迹（突然停止）
        print("\nExample 3: Accident (sudden stop)")
        test_trajectory3 = []
        for i in range(20):
            x1 = 100 + i * 5
            y1 = 100 + i * 2
            x2 = x1 + 50
            y2 = y1 + 30
            test_trajectory3.append([x1, y1, x2, y2])

        # 突然停止
        for i in range(10):
            x1 = 200
            y1 = 140
            x2 = x1 + 50
            y2 = y1 + 30
            test_trajectory3.append([x1, y1, x2, y2])

        pred_class3, confidence3, probs3, explanation3 = predict_single_trajectory(
            model, test_trajectory3, device, feature_scaler, config['max_seq_len']
        )

        print(f"  Prediction: {class_names[pred_class3]}")
        print(f"  Confidence: {confidence3:.2%}")
        print(f"  Probabilities: Normal={probs3[0]:.2%}, Rash={probs3[1]:.2%}, Accident={probs3[2]:.2%}")
        print(f"  Explanation: {explanation3}")

    print("\n" + "=" * 60)
    print("INFERENCE COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()