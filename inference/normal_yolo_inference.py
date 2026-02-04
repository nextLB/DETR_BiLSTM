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

warnings.filterwarnings('ignore')


class YOLOVideoDetector:
    def __init__(self, model_path='yolo11n.pt', video_path=None, conf_threshold=0.25, iou_threshold=0.45):
        """
        初始化YOLO视频检测器

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

        # 颜色映射用于不同类别
        self.colors = plt.cm.tab20(np.linspace(0, 1, 80))

        # 设置matplotlib
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['figure.dpi'] = 100

        # 用于存储视频处理结果
        self.results_queue = Queue(maxsize=10)
        self.processing = False

    def process_video_stream(self):
        """处理视频流"""
        try:
            print(f"Processing video: {self.video_path}")

            # 直接使用YOLO处理视频流
            results = self.model.predict(
                source=self.video_path,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                stream=True,  # 使用流模式
                verbose=False
            )

            self.processing = True
            frame_count = 0

            for result in results:
                if not self.processing:
                    break

                # 获取带标注的帧
                annotated_frame = result.plot(labels=True, line_width=2, font_size=10)

                # 将结果放入队列
                if self.results_queue.full():
                    self.results_queue.get()  # 移除最旧的一帧

                self.results_queue.put({
                    'frame': annotated_frame,
                    'result': result,
                    'frame_num': frame_count
                })

                frame_count += 1
                if frame_count % 30 == 0:  # 每30帧打印一次进度
                    print(f"Processed {frame_count} frames")

            print("Video processing completed")
            self.processing = False

        except Exception as e:
            print(f"Error processing video: {e}")
            import traceback
            traceback.print_exc()
            self.processing = False

    def display_results(self):
        """显示检测结果"""
        if not self.processing:
            print("Starting video processing...")
            # 启动视频处理线程
            self.processing_thread = threading.Thread(target=self.process_video_stream)
            self.processing_thread.daemon = True
            self.processing_thread.start()

            # 等待几秒让队列有数据
            time.sleep(2)

        # 创建图形和轴
        fig, ax = plt.subplots()
        ax.axis('off')
        plt.tight_layout()

        # 添加标题
        plt.suptitle('YOLO Real-time Object Detection', fontsize=16, y=0.95)

        # 添加信息文本
        info_text = ax.text(
            0.02, 0.98,
            'Initializing...',
            transform=ax.transAxes,
            color='white',
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7),
            verticalalignment='top'
        )

        # 初始化图像显示
        img_display = ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))

        # 计时变量
        start_time = time.time()
        fps_counter = 0
        fps = 0
        last_frame_time = time.time()

        def update(frame_num):
            nonlocal fps_counter, fps, start_time, last_frame_time

            try:
                # 从队列获取最新帧
                if not self.results_queue.empty():
                    data = self.results_queue.get_nowait()
                    annotated_frame = data['frame']
                    result = data['result']
                    frame_num = data['frame_num']

                    # 更新图像显示
                    img_display.set_array(annotated_frame)

                    # 设置坐标轴范围
                    ax.set_xlim(0, annotated_frame.shape[1])
                    ax.set_ylim(annotated_frame.shape[0], 0)

                    # 计算FPS
                    current_time = time.time()
                    fps_counter += 1

                    if current_time - start_time >= 1.0:
                        fps = fps_counter / (current_time - start_time)
                        fps_counter = 0
                        start_time = current_time

                    # 更新信息文本
                    total_objects = len(result.boxes) if result.boxes is not None else 0
                    info_text.set_text(f'Frame: {frame_num} | FPS: {fps:.1f} | Objects: {total_objects}')

                    # 显示检测到的类别和数量
                    if total_objects > 0:
                        class_counts = {}
                        for box in result.boxes:
                            cls_id = int(box.cls[0].cpu().numpy())
                            cls_name = self.model.names[cls_id]
                            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

                        # 在图像上显示类别统计
                        stats_text = "\n".join([f"{k}: {v}" for k, v in class_counts.items()])
                        if hasattr(ax, 'stats_text_obj'):
                            ax.stats_text_obj.remove()
                        ax.stats_text_obj = ax.text(
                            0.02, 0.02,
                            stats_text,
                            transform=ax.transAxes,
                            color='white',
                            fontsize=9,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='green', alpha=0.7),
                            verticalalignment='bottom'
                        )

                    last_frame_time = current_time
                    return img_display, info_text
                else:
                    # 队列为空，显示等待信息
                    if time.time() - last_frame_time > 5:  # 5秒没有新帧
                        info_text.set_text('Waiting for frames...')

                    return img_display, info_text

            except Exception as e:
                print(f"Error updating display: {e}")
                return img_display, info_text

        # 设置动画更新间隔（毫秒）
        interval = 33  # 约30FPS

        try:
            from matplotlib.animation import FuncAnimation
            ani = FuncAnimation(
                fig, update,
                frames=None,
                interval=interval,
                blit=True,
                cache_frame_data=False
            )

            # 显示图形
            plt.show()

        except KeyboardInterrupt:
            print("\nDetection interrupted by user.")
        except Exception as e:
            print(f"Error in animation: {e}")
        finally:
            self.processing = False
            if hasattr(self, 'processing_thread'):
                self.processing_thread.join(timeout=1.0)
            print("Display closed.")

    def run_detection(self):
        """运行检测"""
        # 检查视频文件
        video_file = Path(self.video_path)
        if not video_file.exists():
            print(f"Error: Video file not found: {self.video_path}")
            return False

        print(f"Video file found: {self.video_path}")
        print(f"File size: {video_file.stat().st_size / (1024 * 1024):.2f} MB")

        # 开始显示
        self.display_results()
        return True

    def save_detected_video(self, output_path=None):
        """保存检测后的视频"""
        if output_path is None:
            output_path = f"detected_{Path(self.video_path).stem}.mp4"

        print(f"Saving detected video to: {output_path}")

        try:
            results = self.model.predict(
                source=self.video_path,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                save=True,
                project=".",
                name=output_path.replace('.mp4', ''),
                exist_ok=True,
                verbose=True
            )

            print(f"Video saved successfully: {output_path}")
            return True

        except Exception as e:
            print(f"Error saving video: {e}")
            return False


def main():
    # 设置视频路径 - 使用您的实际路径
    video_path = "./home/next_lb/桌面/next/CAR_DETECTION_TRACK/data/test_data/traffic.mp4"

    # 创建检测器
    detector = YOLOVideoDetector(
        model_path='yolo11n.pt',  # 使用YOLOv11n模型
        video_path=video_path,
        conf_threshold=0.3,  # 稍微提高置信度阈值
        iou_threshold=0.45
    )

    # 选择运行模式
    print("\n" + "=" * 60)
    print("YOLO Video Object Detection System")
    print("=" * 60)
    print("\nSelect detection mode:")
    print("1. Real-time display (using matplotlib)")
    print("2. Process and save video")
    print("3. Both display and save")

    try:
        choice = int(input("\nEnter your choice (1-3): "))
    except:
        choice = 1
        print("Using default choice: 1")

    if choice == 1:
        # 实时显示
        print("\nStarting real-time detection display...")
        print("Close the matplotlib window to stop.")
        detector.run_detection()

    elif choice == 2:
        # 处理并保存视频
        output_path = f"detected_{Path(video_path).stem}.mp4"
        print(f"\nProcessing video and saving to: {output_path}")
        detector.save_detected_video(output_path)

    elif choice == 3:
        # 先实时显示，然后保存
        print("\nStarting real-time display...")
        print("Close the matplotlib window when done viewing.")

        # 使用线程同时进行
        import threading

        def run_display():
            detector.run_detection()

        def run_save():
            time.sleep(2)  # 等待显示启动
            output_path = f"detected_{Path(video_path).stem}.mp4"
            print(f"\nProcessing and saving video to: {output_path}")
            detector.save_detected_video(output_path)

        # 启动两个线程
        display_thread = threading.Thread(target=run_display)
        save_thread = threading.Thread(target=run_save)

        display_thread.start()
        save_thread.start()

        display_thread.join()
        save_thread.join()

    else:
        print("Invalid choice. Using default real-time display.")
        detector.run_detection()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback

        traceback.print_exc()