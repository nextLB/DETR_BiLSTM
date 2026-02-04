import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import yaml
import time
from collections import OrderedDict, deque
from typing import List, Dict, Tuple
import warnings

warnings.filterwarnings('ignore')

import sys
sys.path.append('/home/next_lb/桌面/next/CAR_DETECTION_TRACK/')

# 导入自定义模型结构
from train.train_vit_yolo import ViTYOLOModel, VisionTransformer, PatchEmbedding, MultiHeadAttention, TransformerEncoderLayer


class ViTYOLOTracker:
    """使用自定义ViT-YOLO模型进行车辆追踪"""

    def __init__(self, model_path: str, img_size: int = 480, conf_threshold: float = 0.3,
                 iou_threshold: float = 0.5, device: str = None):
        """
        初始化ViT-YOLO追踪器

        Args:
            model_path: 模型权重路径
            img_size: 输入图像尺寸
            conf_threshold: 置信度阈值
            iou_threshold: NMS的IoU阈值
            device: 计算设备 ('cuda' 或 'cpu')
        """
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        print(f"Using device: {self.device}")

        # 加载模型
        self.model = self._load_model(model_path)

        # 追踪相关变量
        self.track_history = {}  # 存储每个track_id的历史位置
        self.track_counter = {}  # 统计每个track_id的出现帧数
        self.next_track_id = 0  # 下一个可用的track_id
        self.max_history_length = 30  # 最多保存的历史轨迹点

        # 为不同track_id预生成颜色
        self.colors = self._generate_colors(100)

        # 类别名称（这里我们只有一个类别：car）
        self.class_names = ['car']

        # FPS计算
        self.fps_buffer = deque(maxlen=30)
        self.prev_time = time.time()

        # 设置matplotlib
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['figure.dpi'] = 100
        plt.ion()  # 启用交互模式

    def _load_model(self, model_path: str):
        """加载训练好的ViT-YOLO模型"""
        print(f"Loading ViT-YOLO model from: {model_path}")

        # 先加载检查点获取配置
        if torch.cuda.is_available():
            checkpoint = torch.load(model_path)
        else:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

        # 从检查点中读取模型配置
        if 'config' in checkpoint:
            config = checkpoint['config']
            print("Loading model configuration from checkpoint:")
            for key, value in config.items():
                print(f"  {key}: {value}")

            # 使用保存的配置创建模型
            model = ViTYOLOModel(
                num_classes=config['num_classes'],
                img_size=config['img_size'],
                patch_size=config['patch_size'],
                embed_dim=config['embed_dim'],
                depth=config['depth'],
                num_heads=config['num_heads']
            )
        else:
            # 如果检查点中没有配置，使用训练时的默认参数
            print("Warning: No configuration found in checkpoint, using default parameters")
            model = ViTYOLOModel(
                num_classes=1,
                img_size=self.img_size,  # 使用推理时的img_size
                patch_size=16,
                embed_dim=512,  # 训练时的默认值
                depth=8,  # 训练时的默认值
                num_heads=8  # 训练时的默认值
            )

        # 加载权重
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(self.device)
        model.eval()
        print("Model loaded successfully!")

        # 显示模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        return model

    def _generate_colors(self, n: int):
        """生成n个不同的颜色"""
        colors = []
        for i in range(n):
            hue = (i * 137) % 180  # 使用黄金角度近似值生成不同色调
            hsv_color = np.ones((1, 1, 3), dtype=np.uint8) * 255
            hsv_color[0, 0, 0] = hue
            hsv_color[0, 0, 1] = 255
            hsv_color[0, 0, 2] = 255
            rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)[0, 0]
            colors.append((rgb_color[0] / 255, rgb_color[1] / 255, rgb_color[2] / 255))
        return colors

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """预处理图像为模型输入格式"""
        # 保持原始图像用于显示
        self.orig_image = image.copy()
        self.orig_h, self.orig_w = image.shape[:2]

        # 转换为RGB并调整大小
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (self.img_size, self.img_size))

        # 归一化并转换为tensor
        image_tensor = torch.from_numpy(image_resized).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

        return image_tensor.to(self.device)

    def decode_predictions(self, predictions: List[torch.Tensor]) -> List[Dict]:
        """解码模型预测结果"""
        detections = []

        # 不同尺度的特征图大小
        feature_map_sizes = [
            self.img_size // 8,  # 浅层特征图
            self.img_size // 16,  # 中层特征图
            self.img_size // 32  # 深层特征图
        ]

        for scale_idx, pred in enumerate(predictions):
            if pred is None:
                continue

            B, C, H, W = pred.shape
            grid_size = feature_map_sizes[scale_idx]

            # 确保预测维度正确 (4+1+1=6)
            if C != 6:
                print(f"Warning: Unexpected prediction channels: {C}, expected 6")
                continue

            # 重塑预测张量
            pred = pred.view(B, 6, H, W).permute(0, 2, 3, 1).contiguous()

            # 提取预测值
            pred_bbox = torch.sigmoid(pred[..., :4])  # [B, H, W, 4]
            pred_obj = torch.sigmoid(pred[..., 4:5])  # [B, H, W, 1]
            pred_cls = torch.sigmoid(pred[..., 5:6])  # [B, H, W, 1]

            # 创建网格坐标
            grid_y, grid_x = torch.meshgrid(
                torch.arange(H, device=self.device),
                torch.arange(W, device=self.device),
                indexing='ij'
            )

            # 转换边界框坐标
            for i in range(H):
                for j in range(W):
                    confidence = pred_obj[0, i, j].item()

                    if confidence > self.conf_threshold:
                        # 获取边界框参数
                        dx, dy, dw, dh = pred_bbox[0, i, j].cpu().numpy()

                        # 转换为绝对坐标 (相对于特征图)
                        x_center = (j + dx) / W
                        y_center = (i + dy) / H
                        width = dw
                        height = dh

                        # 转换到原始图像坐标
                        x1 = int((x_center - width / 2) * self.orig_w)
                        y1 = int((y_center - height / 2) * self.orig_h)
                        x2 = int((x_center + width / 2) * self.orig_w)
                        y2 = int((y_center + height / 2) * self.orig_h)

                        # 确保坐标在图像范围内
                        x1 = max(0, min(self.orig_w - 1, x1))
                        y1 = max(0, min(self.orig_h - 1, y1))
                        x2 = max(0, min(self.orig_w - 1, x2))
                        y2 = max(0, min(self.orig_h - 1, y2))

                        if x2 > x1 and y2 > y1:
                            detections.append({
                                'bbox': [x1, y1, x2, y2],
                                'confidence': confidence,
                                'class_id': 0,
                                'class_name': 'car',
                                'center': [(x1 + x2) // 2, (y1 + y2) // 2]
                            })

        return detections

    def non_max_suppression(self, detections: List[Dict]) -> List[Dict]:
        """应用非极大值抑制"""
        if not detections:
            return []

        # 按置信度排序
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

        boxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['confidence'] for d in detections])

        # 简单的NMS实现
        indices = []
        while len(boxes) > 0:
            # 选择置信度最高的框
            best_idx = np.argmax(scores)
            indices.append(best_idx)
            best_box = boxes[best_idx]

            # 计算与其他所有框的IoU
            x1 = np.maximum(best_box[0], boxes[:, 0])
            y1 = np.maximum(best_box[1], boxes[:, 1])
            x2 = np.minimum(best_box[2], boxes[:, 2])
            y2 = np.minimum(best_box[3], boxes[:, 3])

            inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
            box1_area = (best_box[2] - best_box[0]) * (best_box[3] - best_box[1])
            box2_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            iou = inter_area / (box1_area + box2_area - inter_area + 1e-7)

            # 移除重叠度高的框
            keep_mask = iou < self.iou_threshold
            boxes = boxes[keep_mask]
            scores = scores[keep_mask]

        # 返回筛选后的检测结果
        return [detections[i] for i in indices]

    def update_tracks(self, detections: List[Dict]) -> List[Dict]:
        """更新追踪信息（简单的IOU追踪器）"""
        current_tracks = []

        # 如果没有检测到任何物体，清理旧追踪
        if not detections:
            # 清理太久没有更新的追踪
            tracks_to_remove = []
            for track_id in list(self.track_history.keys()):
                if self.track_counter[track_id] > 30:  # 超过30帧没有更新
                    tracks_to_remove.append(track_id)

            for track_id in tracks_to_remove:
                del self.track_history[track_id]
                del self.track_counter[track_id]

            return []

        # 如果没有现有追踪，为所有检测创建新追踪
        if not self.track_history:
            for det in detections:
                track_id = self.next_track_id
                self.next_track_id += 1

                self.track_history[track_id] = deque(maxlen=self.max_history_length)
                self.track_counter[track_id] = 1

                center = det['center']
                self.track_history[track_id].append(center)

                det['track_id'] = track_id
                current_tracks.append(det)

            return current_tracks

        # 计算检测框与现有追踪的匹配
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(self.track_history.keys())

        # 如果没有任何匹配，尝试根据位置和大小匹配
        if unmatched_tracks and unmatched_detections:
            # 创建IOU矩阵
            iou_matrix = np.zeros((len(unmatched_tracks), len(unmatched_detections)))

            for i, track_id in enumerate(unmatched_tracks):
                # 获取追踪的最后位置
                if self.track_history[track_id]:
                    last_center = self.track_history[track_id][-1]

                    for j, det_idx in enumerate(unmatched_detections):
                        det_center = detections[det_idx]['center']

                        # 计算中心点距离（简化的匹配）
                        distance = np.sqrt((last_center[0] - det_center[0]) ** 2 +
                                           (last_center[1] - det_center[1]) ** 2)

                        # 使用距离的倒数作为匹配分数
                        iou_matrix[i, j] = 1.0 / (distance + 1.0)

            # 进行匹配
            matches = []
            while True:
                if iou_matrix.size == 0:
                    break

                i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                max_value = iou_matrix[i, j]

                if max_value > 0.3:  # 匹配阈值
                    matches.append((unmatched_tracks[i], unmatched_detections[j]))

                    # 移除已匹配的行和列
                    iou_matrix = np.delete(iou_matrix, i, axis=0)
                    iou_matrix = np.delete(iou_matrix, j, axis=1)
                    unmatched_tracks.pop(i)
                    unmatched_detections.pop(j)
                else:
                    break

        # 更新匹配的追踪
        for track_id, det_idx in matches:
            det = detections[det_idx]
            det['track_id'] = track_id

            # 更新追踪历史
            self.track_history[track_id].append(det['center'])
            self.track_counter[track_id] += 1

            current_tracks.append(det)

        # 为未匹配的检测创建新追踪
        for det_idx in unmatched_detections:
            track_id = self.next_track_id
            self.next_track_id += 1

            self.track_history[track_id] = deque(maxlen=self.max_history_length)
            self.track_counter[track_id] = 1

            det = detections[det_idx]
            det['track_id'] = track_id
            self.track_history[track_id].append(det['center'])

            current_tracks.append(det)

        # 清理未匹配的旧追踪（如果太久没有更新）
        for track_id in unmatched_tracks:
            if self.track_counter[track_id] > 5:  # 超过5帧没有匹配
                del self.track_history[track_id]
                del self.track_counter[track_id]

        return current_tracks

    def draw_detections(self, image: np.ndarray, tracks: List[Dict]) -> np.ndarray:
        """在图像上绘制检测框和追踪信息"""
        # 转换为RGB用于matplotlib显示
        display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for track in tracks:
            x1, y1, x2, y2 = track['bbox']
            track_id = track['track_id']
            confidence = track['confidence']

            # 获取该track_id对应的颜色
            color_idx = track_id % len(self.colors)
            color = self.colors[color_idx]
            color_bgr = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))

            # 绘制边界框
            cv2.rectangle(display_image, (x1, y1), (x2, y2), color_bgr, 2)

            # 绘制标签背景
            label = f"ID:{track_id} {track['class_name']} {confidence:.2f}"
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            cv2.rectangle(display_image,
                          (x1, y1 - label_height - baseline - 5),
                          (x1 + label_width, y1),
                          color_bgr, -1)

            # 绘制标签文本
            cv2.putText(display_image, label,
                        (x1, y1 - baseline - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 绘制追踪轨迹
            if track_id in self.track_history and len(self.track_history[track_id]) > 1:
                points = np.array(list(self.track_history[track_id]), dtype=np.int32)

                # 绘制轨迹线
                for i in range(1, len(points)):
                    cv2.line(display_image,
                             tuple(points[i - 1]), tuple(points[i]),
                             color_bgr, 2, lineType=cv2.LINE_AA)

        return display_image

    def add_info_overlay(self, image: np.ndarray, fps: float, frame_count: int) -> np.ndarray:
        """在图像上添加信息覆盖层"""
        h, w = image.shape[:2]

        # 添加半透明背景
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (400, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

        # 添加信息文本（英文）
        texts = [
            "ViT-YOLO Vehicle Tracking System",
            f"Frame: {frame_count}",
            f"FPS: {fps:.1f}",
            f"Vehicles Tracked: {len(self.track_history)}",
            f"Detection Confidence: {self.conf_threshold}",
            f"Model Input Size: {self.img_size}x{self.img_size}"
        ]

        y_offset = 40
        for i, text in enumerate(texts):
            font_size = 0.7 if i > 0 else 0.9
            font_weight = 2 if i == 0 else 1
            color = (255, 255, 255)

            cv2.putText(image, text, (20, y_offset + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, color, font_weight)

        return image

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """处理单帧图像"""
        # 预处理
        input_tensor = self.preprocess_image(frame)

        # 推理
        with torch.no_grad():
            predictions = self.model(input_tensor)

        # 解码预测结果
        detections = self.decode_predictions(predictions)

        # 应用非极大值抑制
        detections = self.non_max_suppression(detections)

        # 更新追踪
        tracks = self.update_tracks(detections)

        # 绘制结果
        result_image = self.draw_detections(self.orig_image.copy(), tracks)

        # 计算FPS
        current_time = time.time()
        fps = 1.0 / (current_time - self.prev_time)
        self.fps_buffer.append(fps)
        self.prev_time = current_time
        avg_fps = np.mean(self.fps_buffer) if self.fps_buffer else fps

        # 添加信息覆盖层
        if hasattr(self, 'frame_count'):
            self.frame_count += 1
        else:
            self.frame_count = 0

        result_image = self.add_info_overlay(result_image, avg_fps, self.frame_count)

        return result_image, tracks

    def process_video(self, video_path: str, output_path: str = None, display: bool = True):
        """处理视频文件"""
        print(f"Processing video: {video_path}")

        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video file: {video_path}")
            return

        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video Info: {width}x{height}, {fps:.2f} FPS, {total_frames} frames")

        # 设置输出视频（如果需要）
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # 初始化matplotlib显示
        if display:
            fig, ax = plt.subplots()
            ax.axis('off')
            plt.tight_layout()

            # 添加标题
            plt.suptitle('ViT-YOLO Real-time Vehicle Tracking', fontsize=16, y=0.95)

            # 初始化图像显示
            img_display = ax.imshow(np.zeros((height, width, 3), dtype=np.uint8))

            # 显示图例
            legend_text = "Legend:\n- Colored Box: Detection with ID\n- Line: Track History\n- ID: Track Identifier"
            legend = ax.text(
                0.98, 0.98, legend_text,
                transform=ax.transAxes,
                color='white',
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7),
                verticalalignment='top',
                horizontalalignment='right'
            )

        frame_count = 0
        processing_times = []

        print("Starting video processing...")
        print("Press 'q' to quit, 'p' to pause")

        paused = False

        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break

                # 处理帧
                start_time = time.time()
                result_image, tracks = self.process_frame(frame)
                processing_time = time.time() - start_time
                processing_times.append(processing_time)

                frame_count += 1

                # 显示进度
                if frame_count % 30 == 0:
                    avg_time = np.mean(processing_times[-30:]) if len(processing_times) > 0 else 0
                    fps_actual = 1.0 / avg_time if avg_time > 0 else 0
                    print(
                        f"Frame {frame_count}/{total_frames} - Processing FPS: {fps_actual:.1f} - Tracks: {len(self.track_history)}")

                # 保存输出视频（如果需要）
                if output_path:
                    # 转换回BGR格式用于视频写入
                    result_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
                    out.write(result_bgr)

                # 显示结果
                if display:
                    img_display.set_data(result_image)
                    ax.set_title(
                        f'Frame: {frame_count} | Vehicles: {len(tracks)} | Active Tracks: {len(self.track_history)}')
                    plt.pause(0.001)

            # 检查按键输入
            if display and plt.fignum_exists(fig.number):
                # 检查是否有按键事件
                if plt.waitforbuttonpress(0.001):
                    key = plt.gcf().canvas.key_press_event.key
                    if key == 'q' or key == 'escape':
                        print("Quitting...")
                        break
                    elif key == 'p':
                        paused = not paused
                        print(f"{'Paused' if paused else 'Resumed'}")
                    elif key == 's':
                        # 保存当前帧
                        save_path = f"frame_{frame_count}.png"
                        cv2.imwrite(save_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
                        print(f"Frame saved to: {save_path}")
            elif not display:
                # 如果没有显示，检查其他退出条件
                if frame_count >= total_frames:
                    break

        # 释放资源
        cap.release()
        if output_path:
            out.release()
            print(f"Output video saved to: {output_path}")

        if display:
            plt.close(fig)

        # 打印统计信息
        print("\n" + "=" * 60)
        print("Processing Statistics:")
        print("=" * 60)
        print(f"Total frames processed: {frame_count}")
        print(f"Average processing time: {np.mean(processing_times) * 1000:.1f} ms")
        print(f"Average FPS: {1.0 / np.mean(processing_times) if np.mean(processing_times) > 0 else 0:.1f}")
        print(f"Total unique vehicles tracked: {self.next_track_id}")
        print(f"Final active tracks: {len(self.track_history)}")

        # 显示最活跃的追踪ID
        if self.track_counter:
            print("\nTop 10 Most Active Tracks:")
            print("-" * 40)
            sorted_tracks = sorted(self.track_counter.items(), key=lambda x: x[1], reverse=True)[:10]
            for track_id, count in sorted_tracks:
                print(f"Track ID {track_id}: {count} frames")

    def process_webcam(self, camera_id: int = 0, display: bool = True):
        """处理摄像头实时视频流"""
        print(f"Starting webcam capture (Camera ID: {camera_id})")

        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Error: Cannot open camera {camera_id}")
            return

        # 设置摄像头参数
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        # 获取实际参数
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"Camera Info: {width}x{height}, {fps:.2f} FPS")

        # 初始化matplotlib显示
        if display:
            fig, ax = plt.subplots()
            ax.axis('off')
            plt.tight_layout()

            # 添加标题
            plt.suptitle('ViT-YOLO Real-time Vehicle Tracking (Webcam)', fontsize=16, y=0.95)

            # 初始化图像显示
            img_display = ax.imshow(np.zeros((height, width, 3), dtype=np.uint8))

            # 显示图例
            legend_text = "Legend:\n- Colored Box: Detection with ID\n- Line: Track History\n- Press 'q' to quit"
            legend = ax.text(
                0.98, 0.98, legend_text,
                transform=ax.transAxes,
                color='white',
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7),
                verticalalignment='top',
                horizontalalignment='right'
            )

        frame_count = 0
        processing_times = []

        print("Starting webcam processing...")
        print("Press 'q' to quit, 'p' to pause")

        paused = False

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame")
                    break

                # 处理帧
                start_time = time.time()
                result_image, tracks = self.process_frame(frame)
                processing_time = time.time() - start_time
                processing_times.append(processing_time)

                frame_count += 1

                # 显示实时FPS
                if frame_count % 10 == 0:
                    avg_time = np.mean(processing_times[-10:]) if len(processing_times) > 0 else 0
                    fps_actual = 1.0 / avg_time if avg_time > 0 else 0
                    if frame_count % 30 == 0:
                        print(f"Frame {frame_count} - FPS: {fps_actual:.1f} - Tracks: {len(self.track_history)}")

                # 显示结果
                if display:
                    img_display.set_data(result_image)
                    ax.set_title(f'Frame: {frame_count} | FPS: {fps_actual:.1f} | Vehicles: {len(tracks)}')
                    plt.pause(0.001)

            # 检查按键输入
            if display and plt.fignum_exists(fig.number):
                # 检查是否有按键事件
                if plt.waitforbuttonpress(0.001):
                    key = plt.gcf().canvas.key_press_event.key
                    if key == 'q' or key == 'escape':
                        print("Quitting...")
                        break
                    elif key == 'p':
                        paused = not paused
                        print(f"{'Paused' if paused else 'Resumed'}")
                    elif key == 's':
                        # 保存当前帧
                        save_path = f"webcam_frame_{frame_count}.png"
                        cv2.imwrite(save_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
                        print(f"Frame saved to: {save_path}")
            else:
                # 如果没有显示窗口，检查其他退出条件
                break

        # 释放资源
        cap.release()

        if display:
            plt.close(fig)

        print(f"\nWebcam processing completed.")
        print(f"Total frames processed: {frame_count}")
        if processing_times:
            print(f"Average FPS: {1.0 / np.mean(processing_times):.1f}")


def main():
    """主函数"""
    print("=" * 60)
    print("ViT-YOLO Vehicle Tracking System")
    print("=" * 60)

    # 设置路径
    dataset_dir = "/home/next_lb/桌面/next/CAR_DETECTION_TRACK"
    model_path = f"{dataset_dir}/train/vit_yolo_checkpoints/checkpoint_epoch_10.pth"

    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        print("Please train the model first using train_vit_yolo.py")
        return

    # 创建追踪器
    print("\nInitializing ViT-YOLO Tracker...")
    tracker = ViTYOLOTracker(
        model_path=model_path,
        img_size=480,  # 与训练时一致
        conf_threshold=0.3,
        iou_threshold=0.5
    )

    # 选择运行模式
    print("\n" + "=" * 60)
    print("Select tracking mode:")
    print("1. Process video file")
    print("2. Real-time webcam tracking")
    print("3. Process video and save output")

    try:
        choice = int(input("\nEnter your choice (1-3): "))
    except:
        choice = 1
        print("Using default choice: 1")

    if choice == 1:
        # 处理视频文件
        video_path = f"{dataset_dir}/data/test_data/traffic.mp4"
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            # 尝试其他可能的位置
            video_path = input("Please enter the full path to the video file: ")

        print(f"\nProcessing video: {video_path}")
        tracker.process_video(video_path, display=True)

    elif choice == 2:
        # 实时摄像头追踪
        camera_id = 0  # 默认摄像头
        use_custom = input("Use default camera (0)? [Y/n]: ").strip().lower()
        if use_custom == 'n':
            try:
                camera_id = int(input("Enter camera ID (usually 0, 1, 2...): "))
            except:
                print("Using default camera ID: 0")

        tracker.process_webcam(camera_id=camera_id, display=True)

    elif choice == 3:
        # 处理视频并保存
        video_path = f"{dataset_dir}/data/test_data/traffic.mp4"
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            video_path = input("Please enter the full path to the video file: ")

        output_path = "vit_yolo_tracked_output.mp4"
        print(f"\nProcessing video: {video_path}")
        print(f"Saving output to: {output_path}")

        tracker.process_video(video_path, output_path=output_path, display=True)

    else:
        print("Invalid choice. Using default: Process video file")
        video_path = f"{dataset_dir}/data/test_data/traffic.mp4"
        if os.path.exists(video_path):
            tracker.process_video(video_path, display=True)
        else:
            print(f"Error: Video file not found: {video_path}")

    print("\n" + "=" * 60)
    print("Program completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        plt.ioff()  # 关闭交互模式
        plt.show()  # 保持窗口打开（如果还有）
