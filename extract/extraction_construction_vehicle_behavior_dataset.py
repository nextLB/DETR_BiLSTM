
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import time
from pathlib import Path
from ultralytics import YOLO
import warnings
import sys
import threading
from queue import Queue
from PIL import Image
import io
import cv2
import os
import json


warnings.filterwarnings('ignore')


class YOLOVehicleTracker:
    def __init__(self, model_path='yolo11n.pt', video_path=None, conf_threshold=0.25, iou_threshold=0.45):
        """
        初始化YOLO车辆追踪器

        Args:
            model_path: YOLO模型路径
            video_path: 视频文件路径
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值
        """
        self.model_path = model_path
        self.video_path = video_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # 初始化模型
        print("Loading YOLO model...")
        self.model = YOLO(model_path)
        print(f"Model loaded: {model_path}")

        # 定义要追踪的车辆类别
        self.vehicle_classes = ['car', 'bus', 'truck']
        self.vehicle_class_ids = self._get_vehicle_class_ids()

        # 颜色映射用于不同追踪ID
        self.track_colors = self._generate_track_colors(100)  # 预生成100个追踪ID的颜色

        # 设置matplotlib
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['figure.dpi'] = 100

        # 用于存储追踪信息
        self.track_history = {}  # 存储每个track_id的历史位置
        self.track_counter = {}  # 统计每个track_id的出现帧数

        # 队列和线程控制
        self.results_queue = Queue(maxsize=10)
        self.processing = False

    def _get_vehicle_class_ids(self):
        """获取车辆类别的ID"""
        vehicle_ids = []
        for vehicle_class in self.vehicle_classes:
            # 查找类名对应的ID
            for idx, name in self.model.names.items():
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
            colors.append((color[0 ] /255, color[1 ] /255, color[2 ] /255))
        return colors

    def _draw_tracking_box(self, image, box, track_id, class_name, confidence):
        """
        在图像上绘制追踪框和标签

        Args:
            image: 输入图像
            box: 边界框 [x1, y1, x2, y2]
            track_id: 追踪ID
            class_name: 类别名称
            confidence: 置信度
        """
        x1, y1, x2, y2 = box

        # 确保坐标在图像范围内
        h, w = image.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))

        # 获取该追踪ID的颜色
        color_idx = track_id % len(self.track_colors)
        color = self.track_colors[color_idx]

        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2),
                      (int(color[0 ] *255), int(color[1 ] *255), int(color[2 ] *255)), 2)

        # 准备标签文本
        label = f"ID:{track_id} {class_name} {confidence:.2f}"

        # 计算标签框大小
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )

        # 绘制标签背景
        cv2.rectangle(image, (x1, y1 - label_height - baseline - 5),
                      (x1 + label_width, y1),
                      (int(color[0 ] *255), int(color[1 ] *255), int(color[2 ] *255)), -1)

        # 绘制标签文本
        cv2.putText(image, label, (x1, y1 - baseline - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return image

    def _draw_tracking_line(self, image, track_history, track_id, color):
        """绘制追踪轨迹线"""
        if track_id in track_history and len(track_history[track_id]) > 1:
            points = np.array(track_history[track_id], dtype=np.int32)

            # 绘制轨迹线
            for i in range(1, len(points)):
                cv2.line(image, tuple(points[ i -1]), tuple(points[i]),
                         (int(color[0 ] *255), int(color[1 ] *255), int(color[2 ] *255)),
                         2, lineType=cv2.LINE_AA)

        return image

    def _update_track_history(self, track_id, center):
        """更新追踪历史"""
        if track_id not in self.track_history:
            self.track_history[track_id] = []
            self.track_counter[track_id] = 0

        self.track_history[track_id].append(center)
        self.track_counter[track_id] += 1

        # 限制历史记录长度
        if len(self.track_history[track_id]) > 30:  # 保留最近30个点
            self.track_history[track_id] = self.track_history[track_id][-30:]

    def process_video_with_tracking(self):
        """处理视频流并进行车辆追踪"""
        try:
            print(f"Processing video with vehicle tracking: {self.video_path}")
            print(f"Tracking classes: {self.vehicle_classes}")

            # 使用YOLO进行追踪（启用tracking模式）
            results = self.model.track(
                source=self.video_path,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=self.vehicle_class_ids,  # 只检测车辆类别
                persist=True,  # 保持追踪ID在帧间一致
                # tracker="bytetrack.yaml",  # 使用ByteTrack追踪器
                tracker="botsort.yaml",  # 使用ByteTrack追踪器
                stream=True,  # 使用流模式
                verbose=False,
            )

            # 获取视频帧的尺寸
            cap = cv2.VideoCapture(self.video_path)
            # 获取基本属性
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 帧宽度
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 帧高度
            fps = cap.get(cv2.CAP_PROP_FPS)  # 帧率 (Frames Per Second)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 总帧数

            print(f"视频尺寸（宽x高）: {width} x {height}")
            print(f"帧率（FPS）: {fps}")
            print(f"总帧数: {frame_count}")
            print(f"时长（秒）: {frame_count / fps:.2f}")

            cap.release()  # 释放资源


            self.processing = True
            frame_count = 0

            # 数据集构建的存储结构
            baseName = os.path.splitext(os.path.basename(self.video_path))[0]
            saveVehicleBehaviorDataPath = f'/home/next_lb/桌面/next/CAR_DETECTION_TRACK/data/BiLSTM_Transformer_data/train_data/{baseName}_extract_data.json'
            vehicleBehaviorDataset = {"0":[-1, -1, -1, -1, -1]}

            for result in results:
                if not self.processing:
                    break

                # 获取原始图像
                original_image = result.orig_img.copy()

                # 获取追踪信息
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes.cpu().numpy()

                    # 检查是否有追踪ID
                    if hasattr(boxes, 'id') and boxes.id is not None:
                        for i, box in enumerate(boxes):
                            # 获取类别信息
                            cls_id = int(box.cls[0])
                            class_name = self.model.names[cls_id]
                            confidence = float(box.conf[0])

                            # 获取边界框坐标
                            x1, y1, x2, y2 = box.xyxy[0]



                            # 获取追踪ID
                            track_id = int(box.id[0])
                            if f"{track_id}" not in vehicleBehaviorDataset:
                                vehicleBehaviorDataset[f"{track_id}"] = [-1, -1, -1, -1, -1]
                            vehicleBehaviorDataset[f"{track_id}"].append([float(x1), float(y1), float(x2), float(y2), int(frame_count)])


                            # 计算中心点用于追踪轨迹
                            center_x = int((x1 + x2) / 2)
                            center_y = int((y1 + y2) / 2)

                            # 更新追踪历史
                            self._update_track_history(track_id, (center_x, center_y))

                            # 绘制追踪框
                            original_image = self._draw_tracking_box(
                                original_image,
                                [x1, y1, x2, y2],
                                track_id,
                                class_name,
                                confidence
                            )

                            # 绘制追踪轨迹
                            color_idx = track_id % len(self.track_colors)
                            color = self.track_colors[color_idx]
                            original_image = self._draw_tracking_line(
                                original_image,
                                self.track_history,
                                track_id,
                                color
                            )

                # 在图像上添加统计信息（英文标注）
                self._add_statistics_overlay(original_image, frame_count, len(boxes) if hasattr(result, 'boxes') and result.boxes is not None else 0)

                # 将结果放入队列
                if self.results_queue.full():
                    self.results_queue.get()  # 移除最旧的一帧

                self.results_queue.put({
                    'frame': cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB),  # 转换为RGB
                    'frame_num': frame_count,
                    'vehicle_count': len(boxes) if hasattr(result, 'boxes') and result.boxes is not None else 0
                })

                frame_count += 1
                if frame_count % 30 == 0:  # 每30帧打印一次进度
                    print(f"Processed {frame_count} frames, Active tracks: {len(self.track_history)}")



            # 保存为JSON文件
            with open(saveVehicleBehaviorDataPath, 'w', encoding='utf-8') as f:
                json.dump(vehicleBehaviorDataset, f, ensure_ascii=False, indent=4)

            print(f"数据已保存到: {saveVehicleBehaviorDataPath}")
            print("Video tracking completed")
            print(f"Total unique vehicles tracked: {len(self.track_counter)}")
            self.processing = False

        except Exception as e:
            print(f"Error processing video: {e}")
            import traceback
            traceback.print_exc()
            self.processing = False

    def _add_statistics_overlay(self, image, frame_num, vehicle_count):
        """在图像上添加统计信息覆盖层"""
        h, w = image.shape[:2]

        # 添加半透明背景
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (350, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

        # 添加统计文本（英文）
        texts = [
            f"Frame: {frame_num}",
            f"Vehicles Detected: {vehicle_count}",
            f"Active Tracks: {len(self.track_history)}",
            f"Vehicle Classes: {', '.join(self.vehicle_classes)}"
        ]

        y_offset = 40
        for i, text in enumerate(texts):
            cv2.putText(image, text, (20, y_offset + i* 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 添加标题
        cv2.putText(image, "YOLO Vehicle Tracking System", (w // 2 - 200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        return image



    def display_tracking_results(self):
        """显示追踪结果"""
        if not self.processing:
            print("Starting vehicle tracking...")
            # 启动视频处理线程
            self.processing_thread = threading.Thread(target=self.process_video_with_tracking)
            self.processing_thread.daemon = True
            self.processing_thread.start()

            # 等待几秒让队列有数据
            time.sleep(2)

        # 从队列获取一帧来获取视频尺寸
        if not self.results_queue.empty():
            data = self.results_queue.get_nowait()
            initial_frame = data['frame']
            # 将帧放回队列以便后续使用
            self.results_queue.put(data)
        else:
            # 如果队列为空，使用默认尺寸
            initial_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # 获取视频尺寸
        height, width = initial_frame.shape[:2]

        # 计算DPI和图形尺寸，保持正确比例
        dpi = 80  # 可以根据需要调整
        fig_width = width / dpi
        fig_height = height / dpi

        # 创建图形和轴，设置正确的尺寸
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
        ax.axis('off')

        # 调整布局，减少空白边距
        plt.tight_layout(pad=0)
        fig.subplots_adjust(left=0, right=1, top=0.95, bottom=0)  # 保留顶部标题空间

        # 添加标题
        plt.suptitle('YOLO Real-time Vehicle Tracking', fontsize=16, y=0.98)

        # 初始化图像显示，使用原始视频尺寸
        img_display = ax.imshow(initial_frame, aspect='auto')

        # 设置坐标轴范围以匹配视频尺寸
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)

        # 添加FPS显示文本
        fps_text = ax.text(
            0.02, 0.98,
            'FPS: Calculating...',
            transform=ax.transAxes,
            color='white',
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='blue', alpha=0.7),
            verticalalignment='top'
        )

        # 添加车辆统计文本
        stats_text = ax.text(
            0.02, 0.02,
            'Vehicles: 0',
            transform=ax.transAxes,
            color='white',
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='green', alpha=0.7),
            verticalalignment='bottom'
        )

        # 计时变量
        frame_times = []
        fps = 0

        def update(frame_num):
            nonlocal fps, frame_times, height, width, fig, ax  # 移到函数开头，包含所有需要修改的外部变量

            try:
                current_time = time.time()

                # 从队列获取最新帧
                if not self.results_queue.empty():
                    # 获取队列中所有帧，只保留最新的一帧
                    while not self.results_queue.empty():
                        data = self.results_queue.get_nowait()

                    # 使用最新的数据
                    frame = data['frame']
                    frame_num = data['frame_num']
                    vehicle_count = data['vehicle_count']

                    # 确保帧尺寸正确
                    if frame.shape[:2] != (height, width):
                        # 如果帧尺寸变化，重新调整图形尺寸
                        height, width = frame.shape[:2]
                        fig.set_size_inches(width / dpi, height / dpi)
                        ax.set_xlim(0, width)
                        ax.set_ylim(height, 0)

                    # 更新图像显示
                    img_display.set_data(frame)

                    # 计算FPS
                    frame_times.append(current_time)
                    # 保留最近10帧的时间
                    frame_times = [t for t in frame_times if current_time - t < 2.0]

                    if len(frame_times) > 1:
                        fps = len(frame_times) / (frame_times[-1] - frame_times[0])

                    # 更新文本
                    fps_text.set_text(f'FPS: {fps:.1f}')
                    stats_text.set_text(f'Vehicles: {vehicle_count}\nActive Tracks: {len(self.track_history)}')

                    # 添加图例说明
                    legend_text = "Legend:\n- Box: Detection\n- ID: Track ID\n- Line: Track History"
                    if hasattr(ax, 'legend_text_obj'):
                        ax.legend_text_obj.remove()
                    ax.legend_text_obj = ax.text(
                        0.98, 0.98,
                        legend_text,
                        transform=ax.transAxes,
                        color='white',
                        fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7),
                        verticalalignment='top',
                        horizontalalignment='right'
                    )

                    return img_display, fps_text, stats_text
                else:
                    # 队列为空，显示等待信息
                    if len(frame_times) > 0 and current_time - frame_times[-1] > 2.0:
                        fps_text.set_text('Processing frames...')

                    return img_display, fps_text, stats_text

            except Exception as e:
                print(f"Error updating display: {e}")
                return img_display, fps_text, stats_text

        # 设置动画
        try:
            from matplotlib.animation import FuncAnimation

            ani = FuncAnimation(
                fig, update,
                frames=None,
                interval=33,  # 约30FPS
                blit=True,
                cache_frame_data=False
            )

            # 显示图形
            plt.show()

        except KeyboardInterrupt:
            print("\nTracking interrupted by user.")
        except Exception as e:
            print(f"Error in animation: {e}")
        finally:
            self.processing = False
            if hasattr(self, 'processing_thread'):
                self.processing_thread.join(timeout=1.0)
            print("Tracking display closed.")



    def run_vehicle_tracking(self):
        """运行车辆追踪"""
        # 检查视频文件
        video_file = Path(self.video_path)
        if not video_file.exists():
            print(f"Error: Video file not found: {self.video_path}")
            return False

        print(f"Video file found: {self.video_path}")
        print(f"File size: {video_file.stat().st_size / (1024 * 1024):.2f} MB")
        print(f"Tracking vehicles: {self.vehicle_classes}")

        # 开始显示追踪结果
        self.display_tracking_results()
        return True

    def save_tracked_video(self, output_path=None):
        """保存追踪后的视频"""
        if output_path is None:
            output_path = f"tracked_{Path(self.video_path).stem}.mp4"

        print(f"Saving tracked video to: {output_path}")
        print(f"Tracking vehicles: {self.vehicle_classes}")

        try:
            # 重置追踪历史
            self.track_history = {}
            self.track_counter = {}

            # 使用YOLO追踪并保存视频
            results = self.model.track(
                source=self.video_path,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=self.vehicle_class_ids,
                persist=True,
                # tracker="bytetrack.yaml",
                tracker="botsort.yaml",  # 使用ByteTrack追踪器
                save=True,
                project=".",
                name=output_path.replace('.mp4', ''),
                exist_ok=True,
                verbose=True
            )

            print(f"Tracked video saved successfully: {output_path}")
            print(f"Total unique vehicles tracked: {len(self.track_counter)}")

            # 打印追踪统计
            print("\nVehicle Tracking Statistics:")
            print("-" * 40)
            for track_id, count in sorted(self.track_counter.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"Track ID {track_id}: {count} frames")

            return True

        except Exception as e:
            print(f"Error saving tracked video: {e}")
            return False


def main():
    # 设置视频路径
    video_path = "/home/next_lb/桌面/next/CAR_DETECTION_TRACK/data/TU-DAT/TU-DAT/Rash-Driving/cctv-v1.mp4"

    # 创建车辆追踪器
    tracker = YOLOVehicleTracker(
        model_path='yolo11x.pt',
        video_path=video_path,
        conf_threshold=0.2,
        iou_threshold=0.45
    )

    # 选择运行模式
    print("\n" + "=" * 60)
    print("YOLO Vehicle Tracking System")
    print("=" * 60)
    print("\nSelect tracking mode:")
    print("1. Real-time vehicle tracking display")
    print("2. Process and save tracked video")
    print("3. Both display and save")

    try:
        choice = int(input("\nEnter your choice (1-3): "))
    except:
        choice = 1
        print("Using default choice: 1")

    if choice == 1:
        # 实时显示追踪
        print("\nStarting real-time vehicle tracking...")
        print("Close the matplotlib window to stop.")
        tracker.run_vehicle_tracking()

    elif choice == 2:
        # 处理并保存追踪视频
        output_path = f"tracked_{Path(video_path).stem}.mp4"
        print(f"\nProcessing video with vehicle tracking and saving to: {output_path}")
        tracker.save_tracked_video(output_path)

    elif choice == 3:
        # 同时进行
        print("\nStarting both real-time display and video saving...")

        def run_display():
            tracker.run_vehicle_tracking()

        def run_save():
            time.sleep(2)
            output_path = f"tracked_{Path(video_path).stem}.mp4"
            print(f"\nProcessing and saving tracked video to: {output_path}")
            tracker.save_tracked_video(output_path)

        # 启动线程
        import threading
        display_thread = threading.Thread(target=run_display)
        save_thread = threading.Thread(target=run_save)

        display_thread.start()
        save_thread.start()

        display_thread.join()
        save_thread.join()

    else:
        print("Invalid choice. Using default real-time tracking.")
        tracker.run_vehicle_tracking()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback

        traceback.print_exc()




