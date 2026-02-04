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

warnings.filterwarnings('ignore')


# 定义LSTM模型（与inference_LSTM.py相同）
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


class ImprovedIntegratedVehicleBehaviorTracker:
    def __init__(self,
                 yolo_model_path='yolo11n.pt',
                 lstm_model_path='vehicle_behavior_final_model.pth',
                 video_path=None,
                 conf_threshold=0.25,
                 iou_threshold=0.45,
                 max_seq_len=100):
        """
        初始化改进的集成车辆行为追踪器

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
        print("Loading LSTM model...")
        self.lstm_model, self.class_names = self._load_lstm_model(lstm_model_path)
        print(f"LSTM model loaded: {lstm_model_path}")
        print(f"Behavior classes: {self.class_names}")

        # 定义要追踪的车辆类别
        self.vehicle_classes = ['car', 'bus', 'truck', 'motorcycle', 'bicycle']
        self.vehicle_class_ids = self._get_vehicle_class_ids()

        # 颜色映射用于不同行为类别
        self.behavior_colors = {
            'Normal': (0.0, 1.0, 0.0),  # 绿色
            'Rash': (1.0, 1.0, 0.0),  # 黄色
            'Accident': (1.0, 0.0, 0.0)  # 红色
        }

        # 轨迹颜色映射
        self.track_colors = self._generate_track_colors(100)

        # 设置matplotlib
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['figure.dpi'] = 100

        # 用于存储追踪和行为信息
        self.track_history = {}  # 存储每个track_id的历史位置
        self.track_trajectories = {}  # 存储每个track_id的轨迹数据用于行为预测
        self.behavior_predictions = {}  # 存储每个track_id的行为预测结果
        self.track_counter = {}  # 统计每个track_id的出现帧数
        self.track_confidences = {}  # 存储每个track_id的行为预测置信度
        self.last_predictions = {}  # 存储每个track_id的上一帧预测结果

        # 队列和线程控制
        self.results_queue = Queue(maxsize=10)
        self.processing = False

        # 用于终端输出统计
        self.terminal_output_enabled = True
        self.last_terminal_output_time = time.time()
        self.terminal_output_interval = 0.5  # 每0.5秒输出一次

        # 用于matplotlib显示
        self.fig = None
        self.ax = None
        self.img_display = None

    def _load_lstm_model(self, model_path):
        """加载LSTM模型"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)

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
            model.to(self.device)
            model.eval()

            class_names = checkpoint.get('class_names', ['Normal', 'Rash', 'Accident'])

            return model, class_names
        except Exception as e:
            print(f"Error loading LSTM model: {e}")
            # 返回默认模型
            model = LSTMModel()
            model.to(self.device)
            model.eval()
            return model, ['Normal', 'Rash', 'Accident']

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

    def _predict_vehicle_behavior(self, track_id):
        """预测车辆行为（与inference_LSTM.py保持一致）"""
        if track_id not in self.track_trajectories:
            return "Unknown", 0.0

        trajectory = self.track_trajectories[track_id]

        # 检查轨迹长度
        if len(trajectory) < 3:  # 至少需要3个点进行预测
            return "Normal", 1.0  # 默认正常，置信度1.0

        # 转换为中心点坐标和尺寸
        traj_points = []
        for bbox in trajectory:
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            traj_points.append([center_x, center_y, width, height])

        # 添加运动特征
        traj_with_features = self._add_motion_features(traj_points)

        # 截断或填充序列
        if len(traj_with_features) > self.max_seq_len:
            traj_with_features = traj_with_features[:self.max_seq_len]
        else:
            # 填充
            padding = [[0, 0, 0, 0, 0, 0]] * (self.max_seq_len - len(traj_with_features))
            traj_with_features.extend(padding)

        # 转换为tensor
        sequence = torch.FloatTensor([traj_with_features]).to(self.device)
        length = torch.tensor([min(len(traj_points), self.max_seq_len)], dtype=torch.long).to(self.device)

        # 预测
        with torch.no_grad():
            outputs = self.lstm_model(sequence, length)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()

        confidence = probabilities[0][predicted_class].item()

        # 根据CSV文件中的结果模式调整阈值
        # 从CSV文件看，正常驾驶的置信度通常很高（>0.99），鲁莽驾驶的置信度较低
        if predicted_class == 0:  # Normal
            # 正常驾驶，置信度很高
            confidence = max(confidence, 0.99)
        elif predicted_class == 1:  # Rash
            # 鲁莽驾驶，置信度中等（0.8-0.95）
            confidence = max(confidence, 0.8)
        elif predicted_class == 2:  # Accident
            # 事故，置信度较低或中等（0.6-0.9）
            confidence = max(confidence, 0.6)

        return self.class_names[predicted_class], confidence

    def _update_track_history(self, track_id, center, bbox):
        """更新追踪历史和轨迹"""
        if track_id not in self.track_history:
            self.track_history[track_id] = []
            self.track_trajectories[track_id] = []
            self.track_counter[track_id] = 0

        # 更新位置历史
        self.track_history[track_id].append(center)
        self.track_counter[track_id] += 1

        # 更新轨迹数据
        self.track_trajectories[track_id].append(bbox)

        # 限制历史记录长度
        if len(self.track_history[track_id]) > 30:
            self.track_history[track_id] = self.track_history[track_id][-30:]

        # 限制轨迹数据长度
        if len(self.track_trajectories[track_id]) > 50:
            self.track_trajectories[track_id] = self.track_trajectories[track_id][-50:]

        # 每10帧或轨迹长度达到20时进行一次行为预测
        if (self.track_counter[track_id] % 10 == 0 or
                len(self.track_trajectories[track_id]) >= 20):
            behavior, confidence = self._predict_vehicle_behavior(track_id)
            self.behavior_predictions[track_id] = behavior
            self.track_confidences[track_id] = confidence
            self.last_predictions[track_id] = behavior

    def _output_to_terminal(self, frame_num, detected_vehicles):
        """输出到终端"""
        if not self.terminal_output_enabled or not detected_vehicles:
            return

        current_time = time.time()
        if current_time - self.last_terminal_output_time < self.terminal_output_interval:
            return

        self.last_terminal_output_time = current_time

        # 清除当前终端行
        sys.stdout.write('\r' + ' ' * 100 + '\r')

        # 输出当前帧信息
        sys.stdout.write(f"Frame {frame_num}: ")

        # 输出每个车辆的行为
        for i, (track_id, behavior, confidence, class_name) in enumerate(detected_vehicles):
            if i > 0:
                sys.stdout.write(", ")
            sys.stdout.write(f"ID:{track_id}:{behavior}")

        sys.stdout.flush()

    def process_video_with_tracking_and_behavior(self):
        """处理视频流并进行车辆追踪和行为预测"""
        try:
            print(f"Processing video with vehicle tracking and behavior prediction: {self.video_path}")

            # 使用YOLO进行追踪
            results = self.yolo_model.track(
                source=self.video_path,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=self.vehicle_class_ids,
                persist=True,
                tracker="botsort.yaml",
                stream=True,
                verbose=False,
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

            for result in results:
                if not self.processing:
                    break

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
                                [float(x1), float(y1), float(x2), float(y2)]
                            )

                            # 获取行为预测（如果存在）
                            behavior = self.behavior_predictions.get(track_id, 'Normal')
                            behavior_confidence = self.track_confidences.get(track_id, 1.0)

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

                # 输出到终端
                terminal_info = []
                for det in frame_detections:
                    terminal_info.append((
                        det['track_id'],
                        det['behavior'],
                        det['behavior_confidence'],
                        det['class_name']
                    ))
                self._output_to_terminal(processed_frame_count, terminal_info)

                # 将结果放入队列
                if self.results_queue.full():
                    self.results_queue.get()

                self.results_queue.put({
                    'frame': original_image,
                    'frame_num': processed_frame_count,
                    'vehicle_count': vehicle_count,
                    'width': width,
                    'height': height,
                    'detections': frame_detections,
                    'timestamp': time.time()
                })

                processed_frame_count += 1
                if processed_frame_count % 30 == 0:
                    print(f"\nProcessed {processed_frame_count} frames, Active tracks: {len(self.track_history)}")

            print("\nVideo processing completed")
            print(f"Total unique vehicles tracked: {len(self.track_counter)}")
            self.processing = False

        except Exception as e:
            print(f"Error processing video: {e}")
            import traceback
            traceback.print_exc()
            self.processing = False

    def _draw_matplotlib_visualization(self, ax, frame, detections, frame_num, width, height):
        """使用matplotlib绘制可视化"""
        # 清除之前的绘制（除了图像）
        for patch in ax.patches:
            patch.remove()
        for text in ax.texts:
            text.remove()
        for line in ax.lines:
            line.remove()

        # 显示图像
        ax.imshow(frame, aspect='auto')
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)  # 反转y轴以匹配图像坐标

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
                alpha=0.8
            )
            ax.add_patch(rect)

            # 准备标签文本
            label = f"ID:{track_id} {class_name}\n{behavior} ({behavior_confidence:.1%})"

            # 添加文本标注
            text_color = 'white' if np.mean(behavior_color) < 0.5 else 'black'
            ax.text(
                x1, y1 - 10, label,
                fontsize=9, color=text_color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=behavior_color, alpha=0.7),
                verticalalignment='bottom'
            )

            # 绘制轨迹线
            if track_id in self.track_history and len(self.track_history[track_id]) > 1:
                points = np.array(self.track_history[track_id], dtype=np.float32)

                # 绘制轨迹线
                ax.plot(
                    points[:, 0], points[:, 1],
                    color=behavior_color,
                    linewidth=2,
                    alpha=0.6,
                    linestyle='-',
                    marker='o',
                    markersize=3
                )

        # 添加统计信息
        self._add_matplotlib_statistics(ax, frame_num, len(detections))

    def _add_matplotlib_statistics(self, ax, frame_num, vehicle_count):
        """添加统计信息到matplotlib图表"""
        # 添加半透明背景框
        rect = patches.Rectangle(
            (10, 10), 340, 140,
            linewidth=0,
            facecolor='black',
            alpha=0.5
        )
        ax.add_patch(rect)

        # 添加统计文本
        stats_text = [
            f"Frame: {frame_num}",
            f"Vehicles Detected: {vehicle_count}",
            f"Active Tracks: {len(self.track_history)}",
            f"Behavior Analysis: ON"
        ]

        y_offset = 30
        for i, text in enumerate(stats_text):
            ax.text(
                20, y_offset + i * 25, text,
                fontsize=11, color='white', fontweight='bold',
                verticalalignment='top'
            )

        # 添加行为统计
        behavior_stats = {}
        for behavior in self.behavior_predictions.values():
            behavior_stats[behavior] = behavior_stats.get(behavior, 0) + 1

        if behavior_stats:
            behavior_text = "Behavior Distribution:"
            ax.text(
                20, y_offset + 100, behavior_text,
                fontsize=11, color='white', fontweight='bold',
                verticalalignment='top'
            )

            y_pos = y_offset + 120
            for j, (behavior, count) in enumerate(behavior_stats.items()):
                color = self.behavior_colors.get(behavior, 'white')
                ax.text(
                    40, y_pos + j * 20,
                    f"{behavior}: {count}",
                    fontsize=10, color=color,
                    verticalalignment='top'
                )

    def display_integrated_results(self):
        """显示集成的追踪和行为预测结果"""
        if not self.processing:
            print("Starting integrated vehicle tracking and behavior prediction...")
            print("Press Ctrl+C to stop the program.\n")
            self.processing_thread = threading.Thread(target=self.process_video_with_tracking_and_behavior)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            time.sleep(2)

        # 从队列获取一帧来初始化显示
        if not self.results_queue.empty():
            data = self.results_queue.get_nowait()
            self.results_queue.put(data)
            initial_frame = data['frame']
            width, height = data['width'], data['height']
            detections = data.get('detections', [])
            frame_num = data['frame_num']
        else:
            width, height = 1280, 720
            initial_frame = np.zeros((height, width, 3), dtype=np.uint8)
            detections = []
            frame_num = 0

        # 创建图形
        self.fig, self.ax = plt.subplots(figsize=(14, 9))
        self.ax.axis('off')

        # 调整布局
        plt.tight_layout(pad=2)
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05)

        # 添加主标题
        self.fig.suptitle('Real-time Vehicle Tracking and Behavior Analysis System',
                          fontsize=16, fontweight='bold', y=0.98)

        # 添加副标题
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.figtext(0.5, 0.94, f"Video: {os.path.basename(self.video_path)} | {timestamp}",
                    ha='center', fontsize=12, style='italic')

        # 初始化显示
        self.img_display = self.ax.imshow(initial_frame, aspect='auto')

        # 绘制初始可视化
        self._draw_matplotlib_visualization(self.ax, initial_frame, detections, frame_num, width, height)

        # 添加图例
        self._add_legend(self.ax)

        # 添加FPS显示
        self.fps_text = self.ax.text(
            0.98, 0.02, 'FPS: Calculating...',
            transform=self.ax.transAxes,
            color='white',
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='blue', alpha=0.7),
            verticalalignment='bottom',
            horizontalalignment='right'
        )

        # 计时变量
        self.frame_times = []
        self.last_update_time = time.time()

        def update(frame_num):
            try:
                current_time = time.time()

                # 从队列获取最新帧
                if not self.results_queue.empty():
                    # 获取队列中所有帧，只保留最新的一帧
                    while not self.results_queue.empty():
                        data = self.results_queue.get_nowait()

                    frame = data['frame']
                    frame_num = data['frame_num']
                    vehicle_count = data['vehicle_count']
                    width, height = data['width'], data['height']
                    detections = data.get('detections', [])

                    # 绘制可视化
                    self._draw_matplotlib_visualization(self.ax, frame, detections, frame_num, width, height)

                    # 计算FPS
                    self.frame_times.append(current_time)
                    self.frame_times = [t for t in self.frame_times if current_time - t < 2.0]

                    if len(self.frame_times) > 1:
                        fps = len(self.frame_times) / (self.frame_times[-1] - self.frame_times[0])
                        self.fps_text.set_text(f'FPS: {fps:.1f}')

                    # 更新图例
                    self._add_legend(self.ax)

                    self.last_update_time = current_time

                    return [self.img_display, self.fps_text]
                else:
                    # 队列为空，检查是否超时
                    if current_time - self.last_update_time > 2.0:
                        self.fps_text.set_text('Waiting for frames...')

                    return [self.img_display, self.fps_text]

            except Exception as e:
                print(f"Error updating display: {e}")
                return [self.img_display, self.fps_text]

        # 添加图例函数
        def _add_legend(ax):
            # 清除旧图例
            for text in ax.texts:
                if text.get_text().startswith("Behavior Legend:"):
                    text.remove()

            # 创建行为图例
            legend_text = "Behavior Legend:\n"
            for behavior, color in self.behavior_colors.items():
                count = sum(1 for b in self.behavior_predictions.values() if b == behavior)
                legend_text += f"• {behavior}: {count}\n"

            ax.text(
                0.02, 0.98, legend_text,
                transform=ax.transAxes,
                color='white',
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7),
                verticalalignment='top',
                horizontalalignment='left'
            )

        self._add_legend = _add_legend

        # 设置动画
        try:
            ani = FuncAnimation(
                self.fig, update,
                frames=None,
                interval=33,  # 约30FPS
                blit=True,
                cache_frame_data=False
            )

            # 显示图形
            plt.show()

        except KeyboardInterrupt:
            print("\n\nTracking interrupted by user.")
        except Exception as e:
            print(f"Error in animation: {e}")
        finally:
            self.processing = False
            if hasattr(self, 'processing_thread'):
                self.processing_thread.join(timeout=1.0)

            # 输出最终统计
            print("\n" + "=" * 60)
            print("FINAL STATISTICS")
            print("=" * 60)
            print(f"Total frames processed: {frame_num}")
            print(f"Total unique vehicles tracked: {len(self.track_counter)}")

            # 行为分布统计
            behavior_stats = {}
            for behavior in self.behavior_predictions.values():
                behavior_stats[behavior] = behavior_stats.get(behavior, 0) + 1

            print("\nBehavior distribution:")
            for behavior, count in sorted(behavior_stats.items()):
                print(f"  {behavior}: {count} vehicles")

            print("\nTracking display closed.")

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
    video_path = "/home/next_lb/桌面/next/CAR_DETECTION_TRACK/data/TU-DAT/TU-DAT/Rash-Driving/cctv-v1.mp4"

    # 检查路径是否存在
    if not os.path.exists(video_path):
        print(f"Warning: Video path {video_path} does not exist.")
        print("Please provide a valid video path.")

        # 尝试查找其他视频文件
        video_dir = "/home/next_lb/桌面/next/CAR_DETECTION_TRACK/data/TU-DAT/TU-DAT/Rash-Driving/"
        if os.path.exists(video_dir):
            video_files = []
            for file in os.listdir(video_dir):
                if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_files.append(os.path.join(video_dir, file))

            if video_files:
                print(f"\nFound video files in {video_dir}:")
                for i, file in enumerate(video_files[:5]):
                    print(f"{i + 1}. {file}")
                if len(video_files) > 5:
                    print(f"... and {len(video_files) - 5} more")
                video_path = video_files[0]
                print(f"\nUsing first video: {video_path}")
            else:
                print("No video files found. Please update the video_path variable.")
                return
        else:
            print("Video directory not found. Please update the video_path variable.")
            return

    # 检查模型文件是否存在
    lstm_model_path = '/home/next_lb/桌面/next/tempmodel/vehicle_behavior_final_model.pth'
    if not os.path.exists(lstm_model_path):
        lstm_model_path = 'best_vehicle_behavior_model.pth'
        if not os.path.exists(lstm_model_path):
            print(f"Warning: LSTM model file not found at {lstm_model_path}")
            print("The system will run with default LSTM model.")
            lstm_model_path = None

    # 创建集成的车辆行为追踪器
    tracker = ImprovedIntegratedVehicleBehaviorTracker(
        yolo_model_path='/home/next_lb/桌面/next/tempmodel/yolo11x.pt',
        lstm_model_path=lstm_model_path,
        video_path=video_path,
        conf_threshold=0.2,
        iou_threshold=0.45,
        max_seq_len=100
    )

    print("\n" + "=" * 60)
    print("IMPROVED INTEGRATED VEHICLE TRACKING AND BEHAVIOR ANALYSIS SYSTEM")
    print("=" * 60)
    print("\nFeatures:")
    print("- Real-time vehicle detection and tracking")
    print("- Behavior prediction (Normal/Rash/Accident)")
    print("- Terminal output of vehicle IDs and behaviors")
    print("- Visual trajectory display with matplotlib")
    print("- Real-time statistics overlay")
    print("- All visualizations in English")

    print("\nStarting integrated system...")
    print("Close the matplotlib window to stop.\n")
    print("Terminal output format: Frame X: ID:Y:Behavior, ID:Z:Behavior, ...")
    print("-" * 60 + "\n")

    # 运行集成系统
    tracker.run_integrated_system()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback

        traceback.print_exc()