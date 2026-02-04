

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

vehicle_classes = ['car', 'bus', 'truck']

def get_vehicle_class_ids(model):
    """获取车辆类别的ID"""
    vehicle_ids = []
    for vehicle_class in vehicle_classes:
        # 查找类名对应的ID
        for idx, name in model.names.items():
            if vehicle_class in name.lower():
                vehicle_ids.append(idx)
    print(f"Tracking vehicle classes: {vehicle_classes}")
    print(f"Corresponding class IDs: {vehicle_ids}")
    return vehicle_ids


def plot_detections(image, boxes, model, track_history):
    """在图像上绘制检测框、ID和轨迹"""
    # 复制图像以避免修改原图
    display_img = image.copy()

    # 为不同ID分配不同的颜色
    colors = plt.cm.get_cmap('hsv', len(boxes) if boxes.id is not None else 1)

    for i, box in enumerate(boxes):
        # 获取类别信息
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        confidence = float(box.conf[0])

        # 获取边界框坐标
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # 获取追踪ID
        track_id = int(box.id[0]) if boxes.id is not None else i

        # 计算中心点
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        # 为当前ID选择颜色
        color = colors(i % len(boxes))
        color_bgr = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))

        # 绘制边界框
        cv2.rectangle(display_img, (x1, y1), (x2, y2), color_bgr, 2)

        # 绘制ID标签背景
        label = f"ID:{track_id} {class_name} {confidence:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(display_img,
                      (x1, y1 - label_size[1] - 10),
                      (x1 + label_size[0], y1),
                      color_bgr, -1)

        # 绘制ID标签
        cv2.putText(display_img, label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 绘制中心点
        cv2.circle(display_img, (center_x, center_y), 4, color_bgr, -1)

        # 绘制轨迹（如果提供了轨迹历史）
        if track_history and track_id in track_history:
            points = track_history[track_id]
            if len(points) > 1:
                # 将点转换为整数
                points = np.array(points, dtype=np.int32)
                # 绘制轨迹线
                for j in range(1, len(points)):
                    cv2.line(display_img,
                             tuple(points[j - 1]),
                             tuple(points[j]),
                             color_bgr, 2)

    return display_img




def main():
    modelPath = '/home/next_lb/桌面/next/tempmodel/yolo11x.pt'
    videoPath = '/home/next_lb/桌面/next/CAR_DETECTION_TRACK/data/TU-DAT/TU-DAT/Rash-Driving/cctv-v1.mp4'
    confThreshold = 0.2
    iouThreshold = 0.45
    model = YOLO(modelPath)
    vehicleClassIds = get_vehicle_class_ids(model)

    results = model.track(
        source=videoPath,
        conf=confThreshold,
        iou=iouThreshold,
        classes=vehicleClassIds,  # 只检测车辆类别
        persist=True,  # 保持追踪ID在帧间一致
        # tracker="bytetrack.yaml",  # 使用ByteTrack追踪器
        tracker="botsort.yaml",  # 使用ByteTrack追踪器
        stream=True,  # 使用流模式
        verbose=False,
    )

    # 获取视频帧的尺寸
    cap = cv2.VideoCapture(videoPath)
    # 获取基本属性
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 帧宽度
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 帧高度
    videoFps = cap.get(cv2.CAP_PROP_FPS)  # 帧率 (Frames Per Second)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 总帧数

    print(f"视频尺寸（宽x高）: {frameWidth} x {frameHeight}")
    print(f"帧率（FPS）: {videoFps}")
    print(f"总帧数: {frameCount}")
    print(f"时长（秒）: {frameCount / videoFps:.2f}")

    cap.release()  # 释放资源

    # 初始化matplotlib图形
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title("YOLO Real-time Vehicle Detection & Tracking", fontsize=16)
    ax.set_xlabel("X-axis (pixels)", fontsize=12)
    ax.set_ylabel("Y-axis (pixels)", fontsize=12)

    # 创建图像显示对象
    imgDisplay = ax.imshow(np.zeros((frameHeight, frameWidth, 3), dtype=np.uint8))
    plt.tight_layout()

    # 存储追踪轨迹
    trackHistory = defaultdict(list)

    # 当前检测的图像帧
    frame_count = 0

    # NOTE: 根据id记录


    for result in results:
        # 获取原始图像
        originalImage = result.orig_img.copy()

        # 更新帧计数器
        frame_count += 1

        # 获取追踪信息
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes.cpu().numpy()

            # 检查是否有追踪ID
            if hasattr(boxes, 'id') and boxes.id is not None:
                for i, box in enumerate(boxes):
                    # 获取类别信息
                    clsId = int(box.cls[0])
                    className = model.names[clsId]
                    confidence = float(box.conf[0])

                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0]

                    # 获取追踪ID
                    trackId = int(box.id[0])

                    # 计算中心点用于追踪轨迹
                    centerX = int((x1 + x2) / 2)
                    centerY = int((y1 + y2) / 2)

                    # 限制轨迹长度
                    if len(trackHistory[trackId]) > 30:
                        trackHistory[trackId].pop(0)
                    trackHistory[trackId].append((centerX, centerY))

            # 绘制检测结果
            displayImg = plot_detections(originalImage, boxes, model, trackHistory)

            # 添加帧信息文本
            infoText = f"Frame: {frame_count}/{frameCount} | Vehicles: {len(boxes)}"
            cv2.putText(displayImg, infoText,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 转换BGR到RGB用于matplotlib显示
            display_img_rgb = cv2.cvtColor(displayImg, cv2.COLOR_BGR2RGB)

            # 更新matplotlib图像
            imgDisplay.set_data(display_img_rgb)
            fig.canvas.draw()
            plt.pause(0.001)  # 短暂暂停以更新显示



    plt.ioff()  # 关闭交互模式
    plt.show()  # 保持窗口打开

if __name__ == '__main__':
    main()




