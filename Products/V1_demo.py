import copy

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np
import time
from pathlib import Path
from ultralytics import YOLO
import warnings
import sys
import threading
from queue import Queue
import cv2
import os
import json
import torch
import torch.nn as nn
from datetime import datetime
from collections import deque
import math

warnings.filterwarnings('ignore')


DISPLACEMENT_THRESHOLD = 0
DISPLACEMENT_COUNT = 10


def euclidean_distance(point1, point2):
    """计算两个点之间的欧几里得距离。

    参数:
    point1: tuple, 第一个点的坐标 (x1, y1)
    point2: tuple, 第二个点的坐标 (x2, y2)

    返回:
    float, 欧几里得距离

    """
    x1 = point1[0]
    y1 = point1[1]
    x2 = point2[0]
    y2 = point2[1]
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


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


class EnhancedVehicleBehaviorTracker:
    def __init__(self,
                 yolo_model_path='yolo11x.pt',
                 lstm_model_path='vehicle_behavior_final_model_improved.pth',
                 video_path=None,
                 conf_threshold=0.25,
                 iou_threshold=0.45,
                 max_seq_len=100):
        """
        初始化增强的车辆行为追踪器

        Args:
            yolo_model_path: YOLO模型路径
            lstm_model_path: LSTM模型路径
            video_path: 视频文件路径
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值
            max_seq_len: LSTM序列最大长度
        """
        self.yolo_model_path = yolo_model_path
        self.lstm_model_path = lstm_model_path
        self.video_path = video_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_seq_len = max_seq_len

        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # 初始化YOLO模型
        print("Loading YOLO model...")
        self.yolo_model = YOLO(yolo_model_path)
        print(f"YOLO model loaded: {yolo_model_path}")

        # 初始化LSTM模型
        print("Loading enhanced LSTM model...")
        self.lstm_model, self.class_names, self.feature_scaler = self._load_lstm_model(lstm_model_path)
        print(f"LSTM model loaded: {lstm_model_path}")
        print(f"Behavior classes: {self.class_names}")
        print(f"Input dimension: {self.lstm_model.input_dim}")

        # 定义要追踪的车辆类别
        self.vehicle_classes = ['car', 'bus', 'truck', 'motorcycle', 'bicycle']
        self.vehicle_class_ids = self._get_vehicle_class_ids()

        # 颜色映射用于不同行为类别
        self.behavior_colors = {
            'Normal': (0.0, 1.0, 0.0),  # 绿色
            'Rash': (1.0, 0.5, 0.0),  # 橙色
            'Accident': (1.0, 0.0, 0.0)  # 红色
        }

        # 轨迹颜色映射
        self.track_colors = self._generate_track_colors(200)

        # 设置matplotlib
        plt.rcParams['figure.figsize'] = [14, 8]
        plt.rcParams['figure.dpi'] = 100

        # 用于存储追踪和行为信息
        self.track_history = {}  # 存储每个track_id的历史位置
        self.track_trajectories = {}  # 存储每个track_id的轨迹数据用于行为预测
        self.behavior_predictions = {}  # 存储每个track_id的行为预测结果
        self.behavior_confidences = {}  # 存储每个track_id的行为预测置信度
        self.behavior_history = {}  # 存储每个track_id的行为历史
        self.track_counter = {}  # 统计每个track_id的出现帧数
        self.last_prediction_frame = {}  # 存储每个track_id上次预测的帧数
        self.lastPredictClass = {}

        # 性能优化
        self.min_trajectory_length = 15  # 最小轨迹长度才开始预测
        self.prediction_interval = 15  # 预测间隔（帧数）
        self.max_history_length = 50  # 最大历史长度

        # 队列和线程控制
        self.results_queue = Queue(maxsize=5)
        self.processing = False
        self.display_active = False

        # 用于终端输出统计
        self.terminal_output_enabled = True
        self.last_terminal_output_time = time.time()
        self.terminal_output_interval = 0.5  # 每0.5秒输出一次

        # 用于matplotlib显示
        self.fig = None
        self.ax = None
        self.last_display_update = 0

        # 性能统计
        self.fps_history = deque(maxlen=30)
        self.processing_times = deque(maxlen=30)
        self.start_time = time.time()
        self.total_frames_processed = 0

    def _load_lstm_model(self, model_path):
        """加载LSTM模型"""
        try:
            if not os.path.exists(model_path):
                # 尝试其他可能的位置
                alt_paths = [
                    model_path,
                    'best_vehicle_behavior_model_improved.pth',
                    'vehicle_behavior_final_model.pth',
                    'best_vehicle_behavior_model.pth'
                ]
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        model_path = alt_path
                        break

            checkpoint = torch.load(model_path, map_location=self.device)

            # 获取配置
            config = checkpoint.get('config', {'hidden_dim': 256, 'num_layers': 3})
            input_dim = checkpoint.get('input_dim', 17)
            feature_scaler = checkpoint.get('feature_scaler', None)
            class_names = checkpoint.get('class_names', ['Normal', 'Rash', 'Accident'])

            # 创建模型
            model = EnhancedLSTMModel(
                input_dim=input_dim,
                hidden_dim=config.get('hidden_dim', 256),
                num_layers=config.get('num_layers', 3),
                num_classes=3,
                dropout=0.0  # 推理时不需要dropout
            )

            # 加载权重
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()

            return model, class_names, feature_scaler
        except Exception as e:
            print(f"Error loading LSTM model: {e}")
            print("Using default model configuration...")
            # 返回默认模型
            model = EnhancedLSTMModel()
            model.to(self.device)
            model.eval()
            return model, ['Normal', 'Rash', 'Accident'], None

    def _get_vehicle_class_ids(self):
        """获取车辆类别的ID"""
        vehicle_ids = []
        for vehicle_class in self.vehicle_classes:
            # 查找类名对应的ID
            for idx, name in self.yolo_model.names.items():
                if vehicle_class in name.lower():
                    vehicle_ids.append(idx)
        print(f"Tracking vehicle classes: {self.vehicle_classes}")
        print(f"Corresponding class IDs: {vehicle_ids}")
        return vehicle_ids

    def _generate_track_colors(self, num_colors):
        """生成用于不同追踪ID的颜色"""
        colors = []
        for i in range(num_colors):
            # 使用HSV色彩空间生成鲜艳的颜色
            hue = (i * 137) % 180  # 使用黄金角度近似值
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2RGB)[0][0]
            colors.append((color[0] / 255, color[1] / 255, color[2] / 255))
        return colors

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

    def _predict_vehicle_behavior(self, track_id):
        """预测车辆行为"""
        if track_id not in self.track_trajectories:
            return "Normal", 0.99, [0.99, 0.01, 0.0]

        trajectory = self.track_trajectories[track_id]

        # 检查轨迹长度
        if len(trajectory) < self.min_trajectory_length:
            return "Normal", 0.99, [0.99, 0.01, 0.0]

        # 转换为中心点坐标和尺寸
        traj_points = []
        for bbox in trajectory:
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            traj_points.append([center_x, center_y, width, height])

        # 添加丰富的运动特征
        traj_with_features = self._add_rich_features(traj_points)

        # 归一化特征
        if self.feature_scaler is not None:
            traj_with_features = self.feature_scaler.transform(traj_with_features)

        # 截断或填充序列
        if len(traj_with_features) > self.max_seq_len:
            traj_with_features = traj_with_features[:self.max_seq_len]
            seq_len = self.max_seq_len
        else:
            padding = [[0] * traj_with_features[0].shape[0]] * (self.max_seq_len - len(traj_with_features))
            traj_with_features.extend(padding)
            seq_len = len(traj_with_features)

        # 转换为tensor
        sequence = torch.FloatTensor([traj_with_features]).to(self.device)
        length = torch.tensor([seq_len], dtype=torch.long).to(self.device)

        # 预测
        with torch.no_grad():
            outputs, _ = self.lstm_model(sequence, length)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()

        confidence = probabilities[0][predicted_class].item()
        all_probabilities = probabilities[0].cpu().numpy()

        if f"{track_id}" not in self.lastPredictClass:
            self.lastPredictClass[f"{track_id}"] = 0

        # NOTE: 加上少量规则限制
        if len(traj_points) >= DISPLACEMENT_COUNT:
            displacementValue = euclidean_distance((traj_points[-DISPLACEMENT_COUNT][0], traj_points[-DISPLACEMENT_COUNT][1]), (traj_points[-1][0], traj_points[-1][1]))
        else:
            displacementValue = 0

        # if displacementValue >= 40

        # 基于简单的位移计算进行简单的距离加成
        if displacementValue >= 30:
            confidence = confidence + displacementValue * 0.001

        print(track_id, predicted_class, confidence, displacementValue)


        if displacementValue <= 50:
            if predicted_class != 0:
                if self.lastPredictClass[f"{track_id}"] != 0:
                    predicted_class = self.lastPredictClass[f"{track_id}"]
                else:
                    predicted_class = 0
        elif displacementValue > 50:
            if self.lastPredictClass[f"{track_id}"] != 0:
                predicted_class = self.lastPredictClass[f"{track_id}"]
            else:
                if predicted_class == 0:
                    predicted_class = 1
                    self.lastPredictClass[f"{track_id}"] = copy.deepcopy(predicted_class)

        # # 应用阈值逻辑
        # if predicted_class == 0:  # Normal
        #     confidence = max(confidence, 0.95)
        # elif predicted_class == 1:  # Rash
        #     if confidence < 0.5:
        #         # 如果置信度不够高，降级为Normal
        #         predicted_class = 0
        #         confidence = 0.95
        #         all_probabilities = np.array([0.95, 0.05, 0.0])
        # elif predicted_class == 2:  # Accident
        #     if confidence < 0.5:
        #         # 如果置信度不够高，降级为Normal
        #         predicted_class = 0
        #         confidence = 0.95
        #         all_probabilities = np.array([0.95, 0.04, 0.01])

        return self.class_names[predicted_class], confidence, all_probabilities

    def _update_track_history(self, track_id, center, bbox, frame_num):
        """更新追踪历史和轨迹"""
        if track_id not in self.track_history:
            self.track_history[track_id] = deque(maxlen=self.max_history_length)
            self.track_trajectories[track_id] = deque(maxlen=self.max_history_length)
            self.behavior_history[track_id] = deque(maxlen=10)
            self.track_counter[track_id] = 0
            self.last_prediction_frame[track_id] = -self.prediction_interval

        # 更新位置历史
        self.track_history[track_id].append(center)
        self.track_counter[track_id] += 1

        # 更新轨迹数据
        self.track_trajectories[track_id].append(bbox)

        # 定期进行行为预测
        frames_since_last_prediction = frame_num - self.last_prediction_frame[track_id]
        if (frames_since_last_prediction >= self.prediction_interval and
                len(self.track_trajectories[track_id]) >= self.min_trajectory_length):

            behavior, confidence, probabilities = self._predict_vehicle_behavior(track_id)
            self.behavior_predictions[track_id] = behavior
            self.behavior_confidences[track_id] = confidence
            self.behavior_history[track_id].append(behavior)
            self.last_prediction_frame[track_id] = frame_num

            # 使用行为历史进行平滑
            if len(self.behavior_history[track_id]) >= 3:
                from collections import Counter
                most_common_behavior = Counter(self.behavior_history[track_id]).most_common(1)[0][0]
                if most_common_behavior != behavior:
                    # 如果最近的行为与当前不一致，使用最常见的行为
                    self.behavior_predictions[track_id] = most_common_behavior
                    self.behavior_confidences[track_id] = 0.9  # 降低置信度

    def _output_to_terminal(self, frame_num, detected_vehicles, fps):
        """输出到终端"""
        if not self.terminal_output_enabled:
            return

        current_time = time.time()
        if current_time - self.last_terminal_output_time < self.terminal_output_interval:
            return

        self.last_terminal_output_time = current_time

        # 清除当前终端行
        sys.stdout.write('\r' + ' ' * 150 + '\r')

        # 输出当前帧信息
        sys.stdout.write(f"Frame {frame_num:5d} | FPS: {fps:5.1f} | ")

        # 输出每个车辆的行为
        for i, (track_id, behavior, confidence, class_name) in enumerate(detected_vehicles):
            if i > 0:
                sys.stdout.write(", ")
            color_code = {
                'Normal': '\033[92m',  # 绿色
                'Rash': '\033[93m',  # 黄色
                'Accident': '\033[91m'  # 红色
            }.get(behavior, '\033[0m')
            sys.stdout.write(f"{color_code}ID:{track_id}:{behavior}({confidence:.0%})\033[0m")

        sys.stdout.flush()

    def process_video_with_tracking_and_behavior(self):
        """处理视频流并进行车辆追踪和行为预测"""
        try:
            print(f"Processing video with enhanced vehicle tracking and behavior prediction: {self.video_path}")

            # 使用YOLO进行追踪
            results = self.yolo_model.track(
                source=self.video_path,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=self.vehicle_class_ids,
                persist=True,
                tracker="bytetrack.yaml",
                stream=True,
                verbose=False,
                imgsz=640,
                half=False
            )

            # 获取视频信息
            cap = cv2.VideoCapture(self.video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            print(f"Video resolution: {width} x {height}")
            print(f"Frame rate: {fps:.2f} FPS")
            print(f"Total frames: {frame_count}")
            print(f"Duration: {frame_count / fps:.2f} seconds")

            cap.release()

            self.processing = True
            processed_frame_count = 0
            last_fps_time = time.time()
            frame_count_since_last = 0

            for result in results:
                if not self.processing:
                    break

                frame_start_time = time.time()

                # 获取原始图像并转换为RGB（用于matplotlib）
                original_image = cv2.cvtColor(result.orig_img.copy(), cv2.COLOR_BGR2RGB)

                # 准备当前帧的检测结果
                frame_detections = []

                # 获取追踪信息
                vehicle_count = 0
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes.cpu().numpy()

                    if hasattr(boxes, 'id') and boxes.id is not None:
                        for i, box in enumerate(boxes):
                            # 获取类别信息
                            cls_id = int(box.cls[0])
                            class_name = self.yolo_model.names[cls_id]
                            confidence = float(box.conf[0])

                            # 获取边界框坐标
                            x1, y1, x2, y2 = box.xyxy[0]

                            # 获取追踪ID
                            track_id = int(box.id[0])

                            # 计算中心点
                            center_x = int((x1 + x2) / 2)
                            center_y = int((y1 + y2) / 2)

                            # 更新追踪历史
                            self._update_track_history(
                                track_id,
                                (center_x, center_y),
                                [float(x1), float(y1), float(x2), float(y2)],
                                processed_frame_count
                            )

                            # 获取行为预测（如果存在）
                            behavior = self.behavior_predictions.get(track_id, 'Normal')
                            behavior_confidence = self.behavior_confidences.get(track_id, 0.99)

                            # 添加到当前帧检测结果
                            frame_detections.append({
                                'track_id': track_id,
                                'bbox': [x1, y1, x2, y2],
                                'class_name': class_name,
                                'confidence': confidence,
                                'behavior': behavior,
                                'behavior_confidence': behavior_confidence,
                                'center': (center_x, center_y)
                            })

                            vehicle_count += 1

                # 计算FPS
                frame_count_since_last += 1
                current_time = time.time()
                if current_time - last_fps_time >= 1.0:
                    fps = frame_count_since_last / (current_time - last_fps_time)
                    self.fps_history.append(fps)
                    frame_count_since_last = 0
                    last_fps_time = current_time

                # 记录处理时间
                processing_time = time.time() - frame_start_time
                self.processing_times.append(processing_time)

                # 输出到终端
                terminal_info = []
                for det in frame_detections:
                    terminal_info.append((
                        det['track_id'],
                        det['behavior'],
                        det['behavior_confidence'],
                        det['class_name']
                    ))

                avg_fps = np.mean(self.fps_history) if self.fps_history else 0
                self._output_to_terminal(processed_frame_count, terminal_info, avg_fps)

                # 将结果放入队列
                if self.results_queue.full():
                    try:
                        self.results_queue.get_nowait()
                    except:
                        pass

                self.results_queue.put({
                    'frame': original_image,
                    'frame_num': processed_frame_count,
                    'vehicle_count': vehicle_count,
                    'width': width,
                    'height': height,
                    'detections': frame_detections,
                    'timestamp': time.time(),
                    'fps': avg_fps
                })

                processed_frame_count += 1
                self.total_frames_processed += 1

                if processed_frame_count % 100 == 0:
                    print(f"\nProcessed {processed_frame_count} frames, "
                          f"Active tracks: {len(self.track_history)}, "
                          f"Avg FPS: {np.mean(self.fps_history) if self.fps_history else 0:.1f}")

            print("\nVideo processing completed")
            print(f"Total unique vehicles tracked: {len(self.track_counter)}")
            self.processing = False

        except Exception as e:
            print(f"Error processing video: {e}")
            import traceback
            traceback.print_exc()
            self.processing = False

    def _draw_visualization(self, frame, detections, frame_num, width, height, fps):
        """绘制可视化"""
        if self.ax is None:
            return

        # 清除之前的绘制
        self.ax.clear()
        self.ax.axis('off')

        # 显示图像
        self.ax.imshow(frame, aspect='auto')
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(height, 0)  # 反转y轴以匹配图像坐标

        # 绘制每个检测到的车辆
        for det in detections:
            track_id = det['track_id']
            bbox = det['bbox']
            class_name = det['class_name']
            behavior = det['behavior']
            behavior_confidence = det['behavior_confidence']
            center = det['center']

            x1, y1, x2, y2 = bbox

            # 根据行为选择颜色
            behavior_color = self.behavior_colors.get(behavior, (0.5, 0.5, 0.5))

            # 创建边界框
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2,
                edgecolor=behavior_color,
                facecolor='none',
                alpha=0.9
            )
            self.ax.add_patch(rect)

            # 准备标签文本
            label = f"ID:{track_id} {class_name}\n{behavior} ({behavior_confidence:.0%})"

            # 添加文本标注
            text_color = 'white' if np.mean(behavior_color) < 0.5 else 'black'
            self.ax.text(
                x1, y1 - 5, label,
                fontsize=8, color=text_color,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=behavior_color, alpha=0.8),
                verticalalignment='bottom'
            )

            # 绘制轨迹线
            if track_id in self.track_history and len(self.track_history[track_id]) > 1:
                points = np.array(self.track_history[track_id], dtype=np.float32)

                # 绘制轨迹线
                line = self.ax.plot(
                    points[:, 0], points[:, 1],
                    color=behavior_color,
                    linewidth=2,
                    alpha=0.7,
                    linestyle='-',
                    marker='',
                )[0]

                # 添加轨迹箭头
                if len(points) > 5:
                    dx = points[-1, 0] - points[-5, 0]
                    dy = points[-1, 1] - points[-5, 1]
                    if dx != 0 or dy != 0:
                        self.ax.arrow(
                            points[-5, 0], points[-5, 1], dx, dy,
                            head_width=10, head_length=15, fc=behavior_color, ec=behavior_color,
                            alpha=0.7
                        )

        # 添加统计信息
        self._add_statistics(frame_num, len(detections), fps)

        # 添加图例
        self._add_legend()

        # 添加标题
        self.fig.suptitle('Enhanced Real-time Vehicle Tracking and Behavior Analysis',
                          fontsize=16, fontweight='bold', y=0.98)

        # 刷新显示
        plt.draw()
        plt.pause(0.001)

    def _add_statistics(self, frame_num, vehicle_count, fps):
        """添加统计信息"""
        if self.ax is None:
            return

        # 添加半透明背景框
        rect = patches.Rectangle(
            (10, 10), 400, 160,
            linewidth=0,
            facecolor='black',
            alpha=0.6
        )
        self.ax.add_patch(rect)

        # 计算行为统计
        behavior_stats = {}
        for behavior in self.behavior_predictions.values():
            behavior_stats[behavior] = behavior_stats.get(behavior, 0) + 1

        # 添加统计文本
        stats_text = [
            f"Frame: {frame_num}",
            f"FPS: {fps:.1f}",
            f"Vehicles Detected: {vehicle_count}",
            f"Active Tracks: {len(self.track_history)}",
            f"Behavior Analysis: ACTIVE"
        ]

        y_offset = 30
        for i, text in enumerate(stats_text):
            self.ax.text(
                20, y_offset + i * 25, text,
                fontsize=11, color='white', fontweight='bold',
                verticalalignment='top'
            )

        # 添加行为统计
        if behavior_stats:
            behavior_text = "Behavior Distribution:"
            self.ax.text(
                20, y_offset + 130, behavior_text,
                fontsize=11, color='white', fontweight='bold',
                verticalalignment='top'
            )

            y_pos = y_offset + 150
            for j, (behavior, count) in enumerate(behavior_stats.items()):
                color = self.behavior_colors.get(behavior, 'white')
                self.ax.text(
                    40, y_pos + j * 20,
                    f"{behavior}: {count}",
                    fontsize=10, color=color,
                    verticalalignment='top'
                )

    def _add_legend(self):
        """添加图例"""
        if self.ax is None:
            return

        # 创建行为图例
        legend_text = "BEHAVIOR LEGEND:\n"
        for behavior, color in self.behavior_colors.items():
            legend_text += f"• {behavior}\n"

        # 添加图例文本
        self.ax.text(
            0.02, 0.98, legend_text,
            transform=self.ax.transAxes,
            color='white',
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7),
            verticalalignment='top',
            horizontalalignment='left'
        )

    def display_integrated_results(self):
        """显示集成的追踪和行为预测结果"""
        if not self.processing:
            print("Starting enhanced vehicle tracking and behavior prediction...")
            print("Press 'q' in the matplotlib window to stop the program.\n")
            self.processing_thread = threading.Thread(target=self.process_video_with_tracking_and_behavior)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            time.sleep(3)

        # 初始化matplotlib图形
        self.fig, self.ax = plt.subplots(figsize=(15, 9))
        plt.tight_layout(pad=2)
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05)

        # 设置关闭事件
        def on_close(event):
            print("\nVisualization window closed by user.")
            self.display_active = False
            self.processing = False

        self.fig.canvas.mpl_connect('close_event', on_close)

        self.display_active = True
        last_frame_num = -1

        try:
            while self.display_active and (self.processing or not self.results_queue.empty()):
                # 检查图形窗口是否仍然打开
                if not plt.fignum_exists(self.fig.number):
                    print("\nVisualization window closed.")
                    self.display_active = False
                    break

                # 检查键盘输入
                if plt.waitforbuttonpress(0.001):
                    key = self.fig.canvas.key_press_event.key
                    if key == 'q' or key == 'escape':
                        print("\nExit requested by user.")
                        self.display_active = False
                        break

                # 从队列获取最新帧
                if not self.results_queue.empty():
                    # 获取队列中所有帧，只保留最新的一帧
                    latest_data = None
                    while not self.results_queue.empty():
                        latest_data = self.results_queue.get_nowait()

                    if latest_data and latest_data['frame_num'] > last_frame_num:
                        # 更新可视化
                        self._draw_visualization(
                            latest_data['frame'],
                            latest_data.get('detections', []),
                            latest_data['frame_num'],
                            latest_data['width'],
                            latest_data['height'],
                            latest_data.get('fps', 0)
                        )
                        last_frame_num = latest_data['frame_num']

                # 短暂暂停以避免过度占用CPU
                plt.pause(0.001)

        except KeyboardInterrupt:
            print("\n\nVisualization interrupted by user.")
        except Exception as e:
            print(f"\nError in visualization: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.display_active = False
            self.processing = False

            # 等待处理线程结束
            if hasattr(self, 'processing_thread'):
                self.processing_thread.join(timeout=3.0)

            # 关闭图形窗口
            if self.fig is not None:
                plt.close(self.fig)

            # 输出最终统计
            self._print_final_statistics()

    def _print_final_statistics(self):
        """打印最终统计信息"""
        print("\n" + "=" * 70)
        print("FINAL STATISTICS")
        print("=" * 70)

        elapsed_time = time.time() - self.start_time
        avg_fps = self.total_frames_processed / elapsed_time if elapsed_time > 0 else 0

        print(f"Total frames processed: {self.total_frames_processed}")
        print(f"Total processing time: {elapsed_time:.2f} seconds")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Total unique vehicles tracked: {len(self.track_counter)}")

        # 行为分布统计
        behavior_stats = {}
        for behavior in self.behavior_predictions.values():
            behavior_stats[behavior] = behavior_stats.get(behavior, 0) + 1

        print("\nBehavior distribution:")
        for behavior, count in sorted(behavior_stats.items()):
            percentage = (count / len(self.behavior_predictions) * 100) if self.behavior_predictions else 0
            print(f"  {behavior}: {count} vehicles ({percentage:.1f}%)")

        # 追踪持续时间统计
        if self.track_counter:
            avg_track_length = np.mean(list(self.track_counter.values()))
            max_track_length = max(self.track_counter.values())
            print(f"\nTracking statistics:")
            print(f"  Average track length: {avg_track_length:.1f} frames")
            print(f"  Maximum track length: {max_track_length} frames")

        print("\nProgram completed successfully!")
        print("=" * 70)

    def run_integrated_system(self):
        """运行集成的追踪和行为分析系统"""
        # 检查视频文件
        video_file = Path(self.video_path)
        if not video_file.exists():
            print(f"Error: Video file not found: {self.video_path}")
            return False

        print(f"Video file found: {self.video_path}")
        print(f"File size: {video_file.stat().st_size / (1024 * 1024):.2f} MB")
        print(f"Tracking vehicle classes: {self.vehicle_classes}")

        # 显示集成结果
        self.display_integrated_results()
        return True


def main():
    """主函数"""
    # 设置视频路径
    video_path = "/home/next_lb/桌面/next/CAR_DETECTION_TRACK/data/TU-DAT/TU-DAT/Final_videos/Positive_Vidoes/v26.mov"
    # 备选视频路径
    alternative_paths = [
        "/home/next_lb/桌面/next/CAR_DETECTION_TRACK/data/TU-DAT/TU-DAT/Rash-Driving/cctv-v1.mp4",
        "/home/next_lb/桌面/next/CAR_DETECTION_TRACK/data/TU-DAT/TU-DAT/Final_videos/Positive_Vidoes/v26.mov",
        "./test_video.mp4"
    ]

    # 检查主路径是否存在
    if not os.path.exists(video_path):
        print(f"Warning: Video path {video_path} does not exist.")
        print("Searching for alternative video files...")

        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                video_path = alt_path
                print(f"Found video: {video_path}")
                break
        else:
            # 尝试在当前目录查找
            video_files = []
            for ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv']:
                video_files.extend(list(Path('.').glob(f'*{ext}')))

            if video_files:
                video_path = str(video_files[0])
                print(f"Using video: {video_path}")
            else:
                print("Error: No video files found!")
                print("Please provide a valid video path.")
                return

    # 检查模型文件
    lstm_model_path = '/home/next_lb/桌面/next/tempmodel/best_vehicle_behavior_model.pth'
    if not os.path.exists(lstm_model_path):
        print(f"Warning: Enhanced LSTM model not found at {lstm_model_path}")
        print("Trying to find alternative model files...")

        alt_models = [
            'best_vehicle_behavior_model_improved.pth',
            '/home/next_lb/桌面/next/tempmodel/best_vehicle_behavior_model.pth',
            'vehicle_behavior_final_model.pth',
            'best_vehicle_behavior_model.pth'
        ]

        for model_path in alt_models:
            if os.path.exists(model_path):
                lstm_model_path = model_path
                print(f"Found model: {lstm_model_path}")
                break
        else:
            print("Warning: No LSTM model files found!")
            print("The system will initialize with default model weights.")
            lstm_model_path = None

    # 创建增强的车辆行为追踪器
    tracker = EnhancedVehicleBehaviorTracker(
        yolo_model_path='/home/next_lb/桌面/next/tempmodel/yolo11x.pt',
        lstm_model_path=lstm_model_path,
        video_path=video_path,
        conf_threshold=0.25,
        iou_threshold=0.45,
        max_seq_len=100
    )

    print("\n" + "=" * 70)
    print("ENHANCED VEHICLE TRACKING AND BEHAVIOR ANALYSIS SYSTEM")
    print("=" * 70)
    print("\nFeatures:")
    print("• Enhanced LSTM model with rich feature engineering")
    print("• Real-time vehicle detection and tracking")
    print("• Behavior prediction (Normal/Rash/Accident) with confidence scores")
    print("• Visual trajectory display with directional arrows")
    print("• Real-time statistics overlay with FPS counter")
    print("• Behavior history smoothing for stable predictions")
    print("• Terminal output with color-coded behavior information")
    print("\nControls:")
    print("• Press 'q' or 'ESC' in the visualization window to exit")
    print("• Close the matplotlib window to stop the program")

    print("\nStarting enhanced system...")
    print("=" * 70 + "\n")

    # 运行集成系统
    try:
        tracker.run_integrated_system()
    except Exception as e:
        print(f"\nError running system: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback

        traceback.print_exc()
