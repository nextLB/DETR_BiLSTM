import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import joblib

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# 1. 定义与训练时相同的模型结构
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

    def _create_mask(self, seq_lens, max_len):
        batch_size = len(seq_lens)
        mask = torch.ones(batch_size, max_len, dtype=torch.bool)
        for i, length in enumerate(seq_lens):
            if length < max_len:
                mask[i, length:] = False
        return mask

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


# 2. 数据预处理类（与训练时相同，但不进行数据增强）
class InferenceTrajectoryDataset(Dataset):
    def __init__(self, data_path, label_path=None, max_seq_len=100):
        self.data_path = data_path
        self.label_path = label_path
        self.max_seq_len = max_seq_len
        self.has_labels = label_path is not None

        # 如果有标签，加载标签
        if self.has_labels:
            self.labels = self._load_labels()
        else:
            self.labels = {}

        # 加载轨迹数据
        self.trajectories, self.labels_list, self.seq_lengths, self.vehicle_ids = self._load_trajectories()

        # 标准化数据
        self.scaler = StandardScaler()
        self._normalize_data()

        print(f"Loaded {len(self.trajectories)} samples")

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

    def _load_trajectories(self):
        trajectories = []
        labels_list = []
        seq_lengths = []
        vehicle_ids = []

        if not os.path.exists(self.data_path):
            print(f"Error: Data file {self.data_path} not found")
            return trajectories, labels_list, seq_lengths, vehicle_ids

        with open(self.data_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error loading JSON: {e}")
                return trajectories, labels_list, seq_lengths, vehicle_ids

        for vehicle_id_str, trajectory in data.items():
            if not trajectory or len(trajectory) < 2:
                continue

            try:
                vehicle_id = int(vehicle_id_str)
            except ValueError:
                continue

            # 获取标签（如果有）
            if self.has_labels:
                label = self.labels.get(vehicle_id, 0)
            else:
                label = -1  # 无标签

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

            trajectories.append(padded_trajectory)
            labels_list.append(label)
            seq_lengths.append(seq_length)
            vehicle_ids.append(vehicle_id)

        return np.array(trajectories), np.array(labels_list), np.array(seq_lengths), np.array(vehicle_ids)

    def _normalize_data(self):
        if len(self.trajectories) > 0:
            original_shape = self.trajectories.shape
            flattened = self.trajectories.reshape(-1, 7)

            non_zero_mask = np.any(flattened != 0, axis=1)
            if non_zero_mask.sum() > 0:
                # 使用训练时的标准化器（需要从训练中保存）
                self.scaler.fit(flattened[non_zero_mask])
                flattened[non_zero_mask] = self.scaler.transform(flattened[non_zero_mask])

            self.trajectories = flattened.reshape(original_shape)

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        trajectory = torch.FloatTensor(self.trajectories[idx])
        label = torch.LongTensor([self.labels_list[idx]]).squeeze()
        seq_len = torch.LongTensor([self.seq_lengths[idx]]).squeeze()
        vehicle_id = self.vehicle_ids[idx]

        return trajectory, label, seq_len, vehicle_id


# 3. 推理函数
def predict(model, dataloader, device):
    model.eval()
    all_predictions = []
    all_labels = []
    all_vehicle_ids = []
    all_probabilities = []

    with torch.no_grad():
        for trajectories, labels, seq_lens, vehicle_ids in tqdm(dataloader, desc="Predicting"):
            trajectories = trajectories.to(device)

            outputs = model(trajectories, seq_lens)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_vehicle_ids.extend(vehicle_ids.numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    return all_predictions, all_labels, all_vehicle_ids, all_probabilities


# 4. 保存结果函数
def save_results(predictions, labels, vehicle_ids, probabilities, output_path):
    results = []

    class_names = {0: "Normal", 1: "Rash", 2: "Accident"}

    for i, (pred, true_label, vid, prob) in enumerate(zip(predictions, labels, vehicle_ids, probabilities)):
        result = {
            "vehicle_id": int(vid),
            "prediction": int(pred),
            "prediction_label": class_names.get(pred, "Unknown"),
            "probability_normal": float(prob[0]),
            "probability_rash": float(prob[1]),
            "probability_accident": float(prob[2]),
        }

        # 如果有真实标签，添加比较
        if true_label != -1:
            result["true_label"] = int(true_label)
            result["true_label_name"] = class_names.get(true_label, "Unknown")
            result["correct"] = bool(pred == true_label)

        results.append(result)

    # 保存为JSON文件
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to {output_path}")

    # 同时保存为CSV格式（可选）
    csv_path = output_path.replace('.json', '.csv')
    with open(csv_path, 'w') as f:
        f.write("vehicle_id,prediction,prediction_label,")
        if labels[0] != -1:
            f.write("true_label,true_label_name,correct,")
        f.write("prob_normal,prob_rash,prob_accident\n")

        for result in results:
            f.write(f"{result['vehicle_id']},{result['prediction']},{result['prediction_label']},")
            if 'true_label' in result:
                f.write(f"{result['true_label']},{result['true_label_name']},{1 if result['correct'] else 0},")
            f.write(
                f"{result['probability_normal']:.4f},{result['probability_rash']:.4f},{result['probability_accident']:.4f}\n")

    print(f"CSV results saved to {csv_path}")

    return results


# 5. 评估函数（如果有真实标签）
def evaluate_results(results):
    if not results or 'true_label' not in results[0]:
        print("\nNo true labels available for evaluation")
        return

    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    accuracy = 100.0 * correct / total if total > 0 else 0

    print(f"\n{'=' * 60}")
    print("EVALUATION RESULTS")
    print(f"{'=' * 60}")
    print(f"Total samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")

    # 类别统计
    class_stats = {0: {'total': 0, 'correct': 0, 'pred_as': {0: 0, 1: 0, 2: 0}},
                   1: {'total': 0, 'correct': 0, 'pred_as': {0: 0, 1: 0, 2: 0}},
                   2: {'total': 0, 'correct': 0, 'pred_as': {0: 0, 1: 0, 2: 0}}}

    class_names = {0: "Normal", 1: "Rash", 2: "Accident"}

    for result in results:
        true_label = result['true_label']
        pred_label = result['prediction']

        class_stats[true_label]['total'] += 1
        class_stats[true_label]['pred_as'][pred_label] += 1

        if result['correct']:
            class_stats[true_label]['correct'] += 1

    print(f"\nClass-wise Performance:")
    print(f"{'-' * 60}")
    print(f"{'Class':<15} {'Total':<10} {'Correct':<10} {'Accuracy':<12} {'Prediction Distribution'}")
    print(f"{'-' * 60}")

    for class_id in range(3):
        stats = class_stats[class_id]
        if stats['total'] > 0:
            acc = 100.0 * stats['correct'] / stats['total']
            pred_dist = f"N:{stats['pred_as'][0]} R:{stats['pred_as'][1]} A:{stats['pred_as'][2]}"
            print(
                f"{class_names[class_id]:<15} {stats['total']:<10} {stats['correct']:<10} {acc:>10.2f}%   {pred_dist}")

    # 混淆矩阵
    print(f"\nConfusion Matrix:")
    print(f"{'':<15} {'Pred Normal':<15} {'Pred Rash':<15} {'Pred Accident':<15}")
    print(f"{'-' * 60}")

    for true_class in range(3):
        row = f"{class_names[true_class]:<15}"
        for pred_class in range(3):
            count = class_stats[true_class]['pred_as'][pred_class]
            row += f"{count:<15}"
        print(row)

    # 错误分析
    print(f"\nError Analysis:")
    errors = [r for r in results if not r['correct']]

    if errors:
        print(f"Total errors: {len(errors)}")
        for error_type in [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]:
            count = sum(1 for r in errors if r['true_label'] == error_type[0] and r['prediction'] == error_type[1])
            if count > 0:
                print(f"  {class_names[error_type[0]]} -> {class_names[error_type[1]]}: {count} errors")

    return accuracy, class_stats


# 6. 主推理函数
def main_inference(data_path, label_path=None, model_path='best_model.pth', config_path='model_config.json'):
    """
    主推理函数

    参数:
    data_path: 数据文件路径
    label_path: 标签文件路径（可选，如果有真实标签）
    model_path: 模型文件路径
    config_path: 配置文件路径
    """

    print(f"\n{'=' * 60}")
    print("BI-LSTM + TRANSFORMER TRAJECTORY PREDICTION")
    print(f"{'=' * 60}")

    # 加载模型配置
    if not os.path.exists(config_path):
        print(f"Error: Config file {config_path} not found")
        return

    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"\nModel Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # 创建模型实例
    model = BiLSTMTransformer(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        dropout=config['dropout'],
        num_classes=config['num_classes']
    ).to(device)

    # 加载模型权重
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found")
        return

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\n✓ Model loaded successfully")
        print(f"  Best validation accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 加载数据
    print(f"\nLoading data from: {data_path}")
    if label_path and os.path.exists(label_path):
        print(f"Loading labels from: {label_path}")
        has_labels = True
    else:
        print("No label file provided, will only output predictions")
        has_labels = False
        label_path = None

    dataset = InferenceTrajectoryDataset(
        data_path=data_path,
        label_path=label_path,
        max_seq_len=100
    )

    if len(dataset) == 0:
        print("Error: No valid data loaded")
        return

    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

    # 进行预测
    predictions, true_labels, vehicle_ids, probabilities = predict(model, dataloader, device)

    # 保存结果
    output_path = 'prediction_results.json'
    results = save_results(predictions, true_labels, vehicle_ids, probabilities, output_path)

    # 打印预测结果
    print(f"\n{'=' * 60}")
    print("PREDICTION RESULTS")
    print(f"{'=' * 60}")

    class_names = {0: "Normal", 1: "Rash", 2: "Accident"}

    # 打印前20个结果
    print(f"\nFirst 20 predictions:")
    print(f"{'-' * 80}")
    print(f"{'Vehicle ID':<12} {'Prediction':<15} {'Probabilities':<40}")
    if has_labels:
        print(f"{'True Label':<15} {'Correct':<10}")
    print(f"{'-' * 80}")

    for i in range(min(100, len(results))):
        result = results[i]
        probs = f"N:{result['probability_normal']:.3f} R:{result['probability_rash']:.3f} A:{result['probability_accident']:.3f}"

        if has_labels:
            correct_str = "✓" if result['correct'] else "✗"
            print(
                f"{result['vehicle_id']:<12} {result['prediction_label']:<15} {probs:<40} {result['true_label_name']:<15} {correct_str:<10}")
        else:
            print(f"{result['vehicle_id']:<12} {result['prediction_label']:<15} {probs:<40}")

    # 统计预测分布
    print(f"\nPrediction Distribution:")
    pred_counts = {0: 0, 1: 0, 2: 0}
    for pred in predictions:
        pred_counts[pred] += 1

    total = len(predictions)
    for class_id in range(3):
        count = pred_counts[class_id]
        percentage = 100.0 * count / total if total > 0 else 0
        print(f"  {class_names[class_id]}: {count} ({percentage:.2f}%)")

    # 如果有真实标签，进行评估
    if has_labels:
        accuracy, class_stats = evaluate_results(results)

        # 保存评估报告
        report = {
            "total_samples": total,
            "accuracy": accuracy,
            "prediction_distribution": pred_counts,
            "class_wise_performance": {}
        }

        for class_id in range(3):
            if class_stats[class_id]['total'] > 0:
                acc = 100.0 * class_stats[class_id]['correct'] / class_stats[class_id]['total']
                report["class_wise_performance"][class_names[class_id]] = {
                    "total": class_stats[class_id]['total'],
                    "correct": class_stats[class_id]['correct'],
                    "accuracy": acc,
                    "predicted_as": class_stats[class_id]['pred_as']
                }

        with open('evaluation_report.json', 'w') as f:
            json.dump(report, f, indent=4)

        print(f"\n✓ Evaluation report saved to evaluation_report.json")

    print(f"\n{'=' * 60}")
    print("INFERENCE COMPLETED")
    print(f"{'=' * 60}")


# 7. 批量推理函数（处理多个文件）
def batch_inference(data_dir, label_dir=None, model_path='best_model.pth', config_path='model_config.json'):
    """
    批量推理函数，处理目录下的所有数据文件

    参数:
    data_dir: 数据文件目录
    label_dir: 标签文件目录（可选）
    model_path: 模型文件路径
    config_path: 配置文件路径
    """

    # 查找所有数据文件
    data_files = [f for f in os.listdir(data_dir) if f.endswith('_extract_data.json')]

    if not data_files:
        print(f"No data files found in {data_dir}")
        return

    print(f"\nFound {len(data_files)} data files for batch inference")

    all_results = []

    # 加载模型
    with open(config_path, 'r') as f:
        config = json.load(f)

    model = BiLSTMTransformer(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        dropout=config['dropout'],
        num_classes=config['num_classes']
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    for data_file in data_files:
        print(f"\nProcessing: {data_file}")

        data_path = os.path.join(data_dir, data_file)

        # 查找对应的标签文件
        if label_dir:
            label_file = data_file.replace('_extract_data.json', '_label.txt')
            label_path = os.path.join(label_dir, label_file)
            if not os.path.exists(label_path):
                print(f"  Warning: Label file {label_file} not found, skipping labels")
                label_path = None
        else:
            label_path = None

        # 加载数据
        dataset = InferenceTrajectoryDataset(
            data_path=data_path,
            label_path=label_path,
            max_seq_len=100
        )

        if len(dataset) == 0:
            print(f"  No valid data in {data_file}")
            continue

        # 进行预测
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
        predictions, true_labels, vehicle_ids, probabilities = predict(model, dataloader, device)

        # 保存单个文件的结果
        results = []
        class_names = {0: "Normal", 1: "Rash", 2: "Accident"}

        for i, (pred, true_label, vid, prob) in enumerate(zip(predictions, true_labels, vehicle_ids, probabilities)):
            result = {
                "data_file": data_file,
                "vehicle_id": int(vid),
                "prediction": int(pred),
                "prediction_label": class_names.get(pred, "Unknown"),
                "probability_normal": float(prob[0]),
                "probability_rash": float(prob[1]),
                "probability_accident": float(prob[2]),
            }

            if true_label != -1:
                result["true_label"] = int(true_label)
                result["true_label_name"] = class_names.get(true_label, "Unknown")
                result["correct"] = bool(pred == true_label)

            results.append(result)
            all_results.append(result)

        # 保存单个文件结果
        output_file = data_file.replace('.json', '_predictions.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)

        print(f"  ✓ Predictions saved to {output_file}")

        # 统计单个文件
        if true_labels[0] != -1:
            correct = sum(1 for r in results if r['correct'])
            accuracy = 100.0 * correct / len(results) if results else 0
            print(f"  Accuracy for this file: {accuracy:.2f}% ({correct}/{len(results)})")

    # 保存所有结果
    if all_results:
        with open('batch_predictions_summary.json', 'w') as f:
            json.dump(all_results, f, indent=4)

        print(f"\n✓ Batch predictions summary saved to batch_predictions_summary.json")

        # 如果有标签，生成总体报告
        if 'true_label' in all_results[0]:
            total = len(all_results)
            correct = sum(1 for r in all_results if r['correct'])
            accuracy = 100.0 * correct / total

            print(f"\n{'=' * 60}")
            print("BATCH INFERENCE SUMMARY")
            print(f"{'=' * 60}")
            print(f"Total samples across all files: {total}")
            print(f"Total correct predictions: {correct}")
            print(f"Overall accuracy: {accuracy:.2f}%")


# 8. 主程序入口
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Trajectory Prediction Inference')
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'batch'],
                        help='Inference mode: single file or batch processing')
    parser.add_argument('--data_path', type=str,
                        help='Path to the data file (for single mode) or data directory (for batch mode)')
    parser.add_argument('--label_path', type=str, default=None,
                        help='Path to the label file (optional, for single mode)')
    parser.add_argument('--label_dir', type=str, default=None,
                        help='Path to the label directory (optional, for batch mode)')
    parser.add_argument('--model_path', type=str, default='best_model.pth',
                        help='Path to the trained model')
    parser.add_argument('--config_path', type=str, default='model_config.json',
                        help='Path to the model configuration')

    args = parser.parse_args()

    if args.mode == 'single':
        if not args.data_path:
            # 如果没有指定路径，使用默认路径
            args.data_path = "/home/next_lb/桌面/next/CAR_DETECTION_TRACK/data/BiLSTM_Transformer_data/train_data/video10_extract_data.json"
            args.label_path = "/home/next_lb/桌面/next/CAR_DETECTION_TRACK/data/BiLSTM_Transformer_data/train_data/video10_label.txt"

        main_inference(
            data_path=args.data_path,
            label_path=args.label_path,
            model_path=args.model_path,
            config_path=args.config_path
        )

    elif args.mode == 'batch':
        if not args.data_path:
            # 使用默认目录
            args.data_path = "/home/next_lb/桌面/next/CAR_DETECTION_TRACK/data/BiLSTM_Transformer_data/train_data/"

        batch_inference(
            data_dir=args.data_path,
            label_dir=args.label_dir,
            model_path=args.model_path,
            config_path=args.config_path
        )



