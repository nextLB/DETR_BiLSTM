import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings

warnings.filterwarnings('ignore')

# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# 定义数据集类（与训练时相同）
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
        self.vehicle_ids = []  # 保存车辆ID

        # 获取所有数据文件
        data_files = [f for f in os.listdir(data_dir) if f.endswith('_extract_data.json')]

        for data_file in data_files:
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
                            # 将标签映射为数字
                            if label_str == 'rash':
                                label = 1
                            elif label_str == 'accident':
                                label = 2
                            else:
                                label = 0
                            label_dict[vehicle_id] = label
            else:
                print(f"Warning: Label file {label_file} not found, using default labels")

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
                self.vehicle_ids.append(f"{data_file}_{vehicle_id}")

        # 转换为numpy数组
        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        self.sequence_lengths = np.array(self.sequence_lengths, dtype=np.int64)

        print(f"Dataset loaded: {len(self.sequences)} sequences")

    def _add_motion_features(self, traj_points):
        """添加运动特征（速度）"""
        traj_with_features = []

        for i in range(len(traj_points)):
            if i == 0:
                # 第一帧，速度为0
                vx, vy = 0, 0
            else:
                # 计算速度（位置变化）
                vx = traj_points[i][0] - traj_points[i - 1][0]
                vy = traj_points[i][1] - traj_points[i - 1][1]

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
            torch.tensor(self.sequence_lengths[idx], dtype=torch.long),
            self.vehicle_ids[idx]
        )


# 定义模型（与训练时相同）
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
            bidirectional=True
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

    def forward(self, x, lengths):
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


def load_model(model_path, device):
    """加载训练好的模型"""
    checkpoint = torch.load(model_path, map_location=device)

    # 获取配置
    config = checkpoint['config']
    input_dim = checkpoint.get('input_dim', 6)

    # 创建模型
    model = LSTMModel(
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

    class_names = checkpoint.get('class_names', ['Normal', 'Rash', 'Accident'])

    print(f"Model loaded from {model_path}")
    print(f"Model configuration: hidden_dim={config['hidden_dim']}, num_layers={config['num_layers']}")

    return model, class_names


def predict_dataset(model, dataloader, device, class_names):
    """对整个数据集进行预测"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_confidences = []
    all_vehicle_ids = []

    with torch.no_grad():
        for batch_idx, (sequences, labels, lengths, vehicle_ids) in enumerate(dataloader):
            sequences = sequences.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            outputs = model(sequences, lengths)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probabilities, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
            all_vehicle_ids.extend(vehicle_ids)

            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx}/{len(dataloader)}")

    return all_predictions, all_labels, all_confidences, all_vehicle_ids


def print_predictions(predictions, labels, confidences, vehicle_ids, class_names, save_to_file=False):
    """打印预测结果"""
    print("\n" + "=" * 80)
    print("PREDICTION RESULTS")
    print("=" * 80)

    correct = 0
    total = len(predictions)

    # 创建结果列表
    results = []
    for i in range(total):
        pred_class = predictions[i]
        true_class = labels[i]
        confidence = confidences[i]
        vehicle_id = vehicle_ids[i]

        is_correct = (pred_class == true_class)
        if is_correct:
            correct += 1

        result = {
            'vehicle_id': vehicle_id,
            'prediction': class_names[pred_class],
            'true_label': class_names[true_class],
            'confidence': float(confidence),
            'correct': is_correct
        }
        results.append(result)

        # 打印前20个结果
        if i < 20:
            status = "✓" if is_correct else "✗"
            print(f"{status} Vehicle: {vehicle_id:<30} Pred: {class_names[pred_class]:<10} "
                  f"True: {class_names[true_class]:<10} Conf: {confidence:.2%}")

    # 打印总体统计
    accuracy = 100 * correct / total
    print("\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    print(f"Total vehicles: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")

    # 按类别统计
    print("\nCLASS-WISE STATISTICS:")
    for i, class_name in enumerate(class_names):
        class_indices = [j for j in range(total) if labels[j] == i]
        if class_indices:
            class_correct = sum(1 for j in class_indices if predictions[j] == labels[j])
            class_accuracy = 100 * class_correct / len(class_indices)
            print(f"  {class_name}: {len(class_indices)} samples, {class_correct} correct ({class_accuracy:.2f}%)")

    # 分类报告
    print("\nDETAILED CLASSIFICATION REPORT:")
    print(classification_report(labels, predictions, target_names=class_names))

    # 混淆矩阵
    print("CONFUSION MATRIX:")
    cm = confusion_matrix(labels, predictions)
    print(cm)

    # 保存到文件
    if save_to_file:
        import csv
        with open('prediction_results.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Vehicle ID', 'Prediction', 'True Label', 'Confidence', 'Correct'])
            for result in results:
                writer.writerow([
                    result['vehicle_id'],
                    result['prediction'],
                    result['true_label'],
                    f"{result['confidence']:.4f}",
                    'Yes' if result['correct'] else 'No'
                ])
        print(f"\nResults saved to 'prediction_results.csv'")


def predict_single_trajectory(model, bbox_sequence, device, max_seq_len=100):
    """预测单个轨迹的行为"""
    model.eval()

    if len(bbox_sequence) < 3:
        return "轨迹太短", 0.0, [0, 0, 0]

    # 转换为中心点坐标和尺寸
    traj_points = []
    for bbox in bbox_sequence:
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        traj_points.append([center_x, center_y, width, height])

    # 添加运动特征
    traj_with_features = []
    for i in range(len(traj_points)):
        if i == 0:
            vx, vy = 0, 0
        else:
            vx = traj_points[i][0] - traj_points[i - 1][0]
            vy = traj_points[i][1] - traj_points[i - 1][1]

        features = [
            traj_points[i][0],  # center_x
            traj_points[i][1],  # center_y
            traj_points[i][2],  # width
            traj_points[i][3],  # height
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
    length = torch.tensor([min(len(traj_points), max_seq_len)], dtype=torch.long).to(device)

    # 预测
    with torch.no_grad():
        outputs = model(sequence, length)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, dim=1).item()

    confidence = probabilities[0][predicted_class].item()
    all_probabilities = probabilities[0].cpu().numpy()

    return predicted_class, confidence, all_probabilities


def main():
    """主函数"""
    # 配置参数（与训练时相同）
    config = {
        'data_dir': '/home/next_lb/桌面/next/CAR_DETECTION_TRACK/data/BiLSTM_Transformer_data/train_data/',
        'label_dir': '/home/next_lb/桌面/next/CAR_DETECTION_TRACK/data/BiLSTM_Transformer_data/train_label/',
        'max_seq_len': 100,
        'batch_size': 32,
    }

    # 选项：测试整个数据集或单个轨迹
    test_mode = 'dataset'  # 'dataset' 或 'single'

    # 加载模型
    model_path = 'vehicle_behavior_final_model.pth'
    if not os.path.exists(model_path):
        model_path = 'best_vehicle_behavior_model.pth'

    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print("Please make sure you have trained the model first.")
        return

    print(f"Loading model from {model_path}...")
    model, class_names = load_model(model_path, device)

    if test_mode == 'dataset':
        # 测试整个数据集
        print("\nLoading dataset for prediction...")
        dataset = VehicleTrajectoryDataset(
            data_dir=config['data_dir'],
            label_dir=config['label_dir'],
            max_seq_len=config['max_seq_len']
        )

        # 创建数据加载器
        dataloader = DataLoader(
            dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2
        )

        # 进行预测
        print("Making predictions on the entire dataset...")
        predictions, labels, confidences, vehicle_ids = predict_dataset(
            model, dataloader, device, class_names
        )

        # 打印结果
        print_predictions(predictions, labels, confidences, vehicle_ids, class_names, save_to_file=True)

    elif test_mode == 'single':
        # 测试单个轨迹
        print("\nTesting single trajectory prediction...")

        # 创建一个测试轨迹（模拟数据）
        # 这里可以替换为您的实际轨迹数据
        test_trajectory = []

        # 模拟一个正常行驶的车辆轨迹（缓慢移动）
        print("Example 1: Normal driving (slow movement)")
        for i in range(20):
            x1 = 100 + i * 2  # 缓慢向右移动
            y1 = 100 + i * 1  # 缓慢向下移动
            x2 = x1 + 50
            y2 = y1 + 30
            test_trajectory.append([x1, y1, x2, y2])

        pred_class, confidence, probs = predict_single_trajectory(
            model, test_trajectory, device, config['max_seq_len']
        )

        print(f"Prediction: {class_names[pred_class]}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Probabilities: Normal={probs[0]:.2%}, Rash={probs[1]:.2%}, Accident={probs[2]:.2%}")

        # 模拟一个rash driving的轨迹（快速移动）
        print("\nExample 2: Rash driving (fast movement)")
        test_trajectory2 = []
        for i in range(20):
            x1 = 100 + i * 10  # 快速向右移动
            y1 = 100 + i * 5  # 快速向下移动
            x2 = x1 + 50
            y2 = y1 + 30
            test_trajectory2.append([x1, y1, x2, y2])

        pred_class2, confidence2, probs2 = predict_single_trajectory(
            model, test_trajectory2, device, config['max_seq_len']
        )

        print(f"Prediction: {class_names[pred_class2]}")
        print(f"Confidence: {confidence2:.2%}")
        print(f"Probabilities: Normal={probs2[0]:.2%}, Rash={probs2[1]:.2%}, Accident={probs2[2]:.2%}")

    print("\nPrediction completed successfully!")


if __name__ == "__main__":
    main()


# import json
# import os
# import numpy as np
# import torch
# import torch.nn as nn
# import pandas as pd
# from torch.utils.data import Dataset, DataLoader
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# import warnings
# import csv
# from collections import defaultdict
#
# warnings.filterwarnings('ignore')
#
# # 设置设备
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")
#
#
# class InferenceLSTMModel(nn.Module):
#     """推理用的LSTM模型，结构与训练时一致"""
#
#     def __init__(self, input_dim=6, hidden_dim=128, num_layers=2, num_classes=3, dropout=0.5):
#         super(InferenceLSTMModel, self).__init__()
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
#     def forward(self, x, lengths):
#         packed_input = nn.utils.rnn.pack_padded_sequence(
#             x, lengths.cpu(), batch_first=True, enforce_sorted=False
#         )
#
#         packed_output, (hidden, cell) = self.lstm(packed_input)
#         output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
#
#         output = output.permute(0, 2, 1)
#         output = self.bn(output)
#         output = output.permute(0, 2, 1)
#
#         attention_weights = torch.softmax(self.attention(output), dim=1)
#         context_vector = torch.sum(attention_weights * output, dim=1)
#
#         logits = self.classifier(context_vector)
#         return logits
#
#
# class InferenceDataset(Dataset):
#     """推理数据集类"""
#
#     def __init__(self, data_dir, label_dir, max_seq_len=100):
#         self.data_dir = data_dir
#         self.label_dir = label_dir
#         self.max_seq_len = max_seq_len
#         self.sequences = []
#         self.labels = []
#         self.sequence_lengths = []
#         self.identifiers = []  # 存储文件名和车辆ID用于标识
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
#                 # 添加运动特征
#                 traj_with_features = self._add_motion_features(traj_points)
#
#                 # 截断或填充序列
#                 if len(traj_with_features) > self.max_seq_len:
#                     traj_with_features = traj_with_features[:self.max_seq_len]
#                     seq_len = self.max_seq_len
#                 else:
#                     padding = [[0, 0, 0, 0, 0, 0]] * (self.max_seq_len - len(traj_with_features))
#                     traj_with_features.extend(padding)
#                     seq_len = len(traj_points)
#
#                 self.sequences.append(traj_with_features)
#                 self.labels.append(label)
#                 self.sequence_lengths.append(seq_len)
#                 self.identifiers.append(f"{base_name}_{vehicle_id}")
#
#         # 转换为numpy数组
#         self.sequences = np.array(self.sequences, dtype=np.float32)
#         self.labels = np.array(self.labels, dtype=np.int64)
#         self.sequence_lengths = np.array(self.sequence_lengths, dtype=np.int64)
#
#         print(f"Inference dataset loaded: {len(self.sequences)} sequences")
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
#             torch.tensor(self.sequence_lengths[idx], dtype=torch.long),
#             self.identifiers[idx]
#         )
#
#
# def load_model(model_path, device):
#     """加载训练好的模型"""
#     # 加载模型检查点
#     checkpoint = torch.load(model_path, map_location=device)
#
#     # 获取配置
#     config = checkpoint.get('config', {})
#     input_dim = checkpoint.get('input_dim', 6)
#     thresholds = checkpoint.get('thresholds', [0.3, 0.3])
#
#     # 创建模型
#     model = InferenceLSTMModel(
#         input_dim=input_dim,
#         hidden_dim=config.get('hidden_dim', 64),
#         num_layers=config.get('num_layers', 2),
#         num_classes=3,
#         dropout=config.get('dropout', 0.5)
#     )
#
#     # 加载模型权重
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model = model.to(device)
#     model.eval()
#
#     print(f"Model loaded from {model_path}")
#     print(f"Model configuration: {config}")
#     print(f"Thresholds for minority classes: {thresholds}")
#
#     return model, thresholds
#
#
# def predict(model, dataloader, device, thresholds=None):
#     """进行批量预测"""
#     predictions = []
#     true_labels = []
#     probabilities = []
#     identifiers = []
#
#     with torch.no_grad():
#         for batch in dataloader:
#             sequences, labels, lengths, batch_ids = batch
#             sequences = sequences.to(device)
#             lengths = lengths.to(device)
#
#             # 获取模型输出
#             outputs = model(sequences, lengths)
#             batch_probs = torch.softmax(outputs, dim=1).cpu().numpy()
#
#             # 根据阈值进行预测
#             if thresholds:
#                 batch_preds = []
#                 for probs in batch_probs:
#                     if probs[1] > thresholds[0]:  # Rash阈值
#                         batch_preds.append(1)
#                     elif probs[2] > thresholds[1]:  # Accident阈值
#                         batch_preds.append(2)
#                     else:
#                         batch_preds.append(0)
#             else:
#                 # 使用argmax
#                 batch_preds = torch.argmax(outputs, dim=1).cpu().numpy()
#
#             # 收集结果
#             predictions.extend(batch_preds)
#             true_labels.extend(labels.cpu().numpy())
#             probabilities.extend(batch_probs)
#             identifiers.extend(batch_ids)
#
#     return predictions, true_labels, probabilities, identifiers
#
#
# def save_results_to_csv(results, output_file):
#     """将结果保存到CSV文件"""
#     with open(output_file, 'w', newline='', encoding='utf-8') as f:
#         writer = csv.writer(f)
#
#         # 写入表头
#         header = ['Identifier', 'True_Label', 'Predicted_Label', 'Normal_Prob',
#                   'Rash_Prob', 'Accident_Prob', 'Is_Correct']
#         writer.writerow(header)
#
#         # 写入数据
#         for i in range(len(results['identifiers'])):
#             identifier = results['identifiers'][i]
#             true_label = results['true_labels'][i]
#             pred_label = results['predictions'][i]
#             probs = results['probabilities'][i]
#
#             # 标签映射
#             label_names = {0: 'Normal', 1: 'Rash', 2: 'Accident'}
#             true_label_name = label_names.get(true_label, 'Unknown')
#             pred_label_name = label_names.get(pred_label, 'Unknown')
#
#             row = [
#                 identifier,
#                 true_label_name,
#                 pred_label_name,
#                 f"{probs[0]:.4f}",
#                 f"{probs[1]:.4f}",
#                 f"{probs[2]:.4f}",
#                 'Yes' if true_label == pred_label else 'No'
#             ]
#             writer.writerow(row)
#
#     print(f"Results saved to {output_file}")
#
#
# def calculate_metrics(true_labels, predictions, label_names=['Normal', 'Rash', 'Accident']):
#     """计算评估指标"""
#     print("\n" + "=" * 60)
#     print("模型性能评估结果")
#     print("=" * 60)
#
#     # 计算准确率
#     accuracy = accuracy_score(true_labels, predictions)
#     print(f"整体准确率: {accuracy:.4f}")
#
#     # 分类报告
#     print("\n分类报告:")
#     print(classification_report(true_labels, predictions, target_names=label_names))
#
#     # 混淆矩阵
#     cm = confusion_matrix(true_labels, predictions)
#     print("混淆矩阵:")
#     print(cm)
#
#     # 各类别准确率
#     print("\n各类别准确率:")
#     for i, name in enumerate(label_names):
#         idx = np.where(np.array(true_labels) == i)[0]
#         if len(idx) > 0:
#             class_acc = np.mean(np.array(predictions)[idx] == i)
#             print(f"  {name}: {class_acc:.4f} ({len(idx)}个样本)")
#
#     return {
#         'accuracy': accuracy,
#         'confusion_matrix': cm,
#         'classification_report': classification_report(true_labels, predictions, target_names=label_names,
#                                                        output_dict=True)
#     }
#
#
# def main():
#     """主推理函数"""
#     # 配置参数
#     model_path = 'improved_final_model.pth'  # 训练好的模型路径
#     data_dir = '/home/next_lb/桌面/next/CAR_DETECTION_TRACK/data/BiLSTM_Transformer_data/train_data/'  # 数据目录
#     label_dir = '/home/next_lb/桌面/next/CAR_DETECTION_TRACK/data/BiLSTM_Transformer_data/train_label/'  # 标签目录
#     output_csv = 'inference_results.csv'  # 输出CSV文件
#
#     # 加载模型
#     print("加载模型中...")
#     model, thresholds = load_model(model_path, device)
#
#     # 创建数据集
#     print("创建推理数据集...")
#     dataset = InferenceDataset(
#         data_dir=data_dir,
#         label_dir=label_dir,
#         max_seq_len=50  # 与训练时一致
#     )
#
#     # 创建数据加载器
#     dataloader = DataLoader(
#         dataset,
#         batch_size=16,
#         shuffle=False,
#         num_workers=2
#     )
#
#     # 进行预测
#     print("进行推理预测...")
#     predictions, true_labels, probabilities, identifiers = predict(
#         model, dataloader, device, thresholds
#     )
#
#     # 计算评估指标
#     metrics = calculate_metrics(true_labels, predictions)
#
#     # 准备结果字典
#     results = {
#         'identifiers': identifiers,
#         'true_labels': true_labels,
#         'predictions': predictions,
#         'probabilities': probabilities
#     }
#
#     # 保存结果到CSV
#     save_results_to_csv(results, output_csv)
#
#     # 额外保存一个汇总报告
#     summary_file = 'inference_summary.txt'
#     with open(summary_file, 'w', encoding='utf-8') as f:
#         f.write("推理结果汇总报告\n")
#         f.write("=" * 50 + "\n\n")
#         f.write(f"总样本数: {len(predictions)}\n")
#         f.write(f"整体准确率: {metrics['accuracy']:.4f}\n\n")
#
#         # 各类别统计
#         f.write("各类别统计:\n")
#         for i, name in enumerate(['Normal', 'Rash', 'Accident']):
#             true_count = sum(1 for label in true_labels if label == i)
#             pred_count = sum(1 for pred in predictions if pred == i)
#             correct_count = sum(1 for j in range(len(true_labels))
#                                 if true_labels[j] == i and predictions[j] == i)
#
#             f.write(f"  {name}:\n")
#             f.write(f"    真实数量: {true_count}\n")
#             f.write(f"    预测数量: {pred_count}\n")
#             f.write(f"    正确预测: {correct_count}\n")
#             if true_count > 0:
#                 f.write(f"    类别准确率: {correct_count / true_count:.4f}\n")
#             f.write("\n")
#
#         # 混淆矩阵
#         f.write("混淆矩阵:\n")
#         cm_str = str(metrics['confusion_matrix'])
#         f.write(cm_str + "\n")
#
#     print(f"\n详细汇总报告已保存到: {summary_file}")
#
#     # 打印一些样本的预测结果
#     print("\n" + "=" * 60)
#     print("样本预测示例:")
#     print("=" * 60)
#     for i in range(min(10, len(predictions))):
#         label_names = {0: 'Normal', 1: 'Rash', 2: 'Accident'}
#         true_name = label_names.get(true_labels[i], 'Unknown')
#         pred_name = label_names.get(predictions[i], 'Unknown')
#
#         print(f"样本 {i + 1}: {identifiers[i]}")
#         print(f"  真实标签: {true_name}")
#         print(f"  预测标签: {pred_name}")
#         print(f"  概率分布: Normal={probabilities[i][0]:.3f}, "
#               f"Rash={probabilities[i][1]:.3f}, "
#               f"Accident={probabilities[i][2]:.3f}")
#         print(f"  是否正确: {'是' if true_labels[i] == predictions[i] else '否'}")
#         print()
#
#
# if __name__ == "__main__":
#     main()

