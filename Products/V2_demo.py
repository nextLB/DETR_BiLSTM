import copy

from ultralytics import YOLO
import os
import threading
from queue import Queue
import time
import cv2
import sys
sys.path.append('/home/next_lb/桌面/next/DETR_BiLSTM')
from BiLSTM_TRS_MODEL.V2 import BiLSTMTransformer
from collections import defaultdict
import numpy as np
import re
import json

EXTRACT_BILSTM_RAW_DATA_PATH = '/home/next_lb/桌面/next/CAR_DETECTION_TRACK/data/V2_BiLSTM_DATA/train_data/'

def draw_detections_with_trajectories(image_data, history_basic_info,
                                      history_points=20,  # 要绘制的历史轨迹点数
                                      current_video_name=None,
                                      current_frame_count=None):
    """
    在图像上绘制车辆检测框、追踪ID、历史轨迹和当前位置

    参数:
        image_data: 当前帧图像数据
        history_basic_info: 历史检测信息列表，每个元素为(video_name, class_name, track_id, x1, y1, x2, y2, confidence, frame_count)
        history_points: 要绘制的历史轨迹点数
        current_video_name: 当前视频名称（可选，用于筛选特定视频的数据）
        current_frame_count: 当前帧计数（可选，用于筛选特定帧的数据）

    返回:
        绘制后的图像
    """
    # 创建一个原图像的副本，避免修改原图像
    img = image_data.copy()

    # 如果提供了高度和宽度参数，则调整图像大小
    if isinstance(image_data, np.ndarray):
        height, width = img.shape[:2]
    else:
        height, width = 1920, 1080  # 默认值

    # 按追踪ID组织历史数据
    track_history = defaultdict(list)

    # 筛选并组织数据
    for info in history_basic_info:
        video_name, class_name, track_id, x1, y1, x2, y2, conf, frame_count = info

        # 如果指定了视频名称，只处理该视频的数据
        if current_video_name is not None and video_name != current_video_name:
            continue

        # 如果指定了帧计数，只处理该帧及之前的数据
        if current_frame_count is not None and frame_count > current_frame_count:
            continue

        # 计算边界框中心点
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # 存储到对应追踪ID的历史中
        track_history[track_id].append({
            'frame_count': frame_count,
            'center': (center_x, center_y),
            'bbox': (x1, y1, x2, y2),
            'class_name': class_name,
            'confidence': conf
        })

    # 为每个追踪ID分配一个颜色（使用HSV颜色空间均匀分布）
    colors = {}
    track_ids = list(track_history.keys())
    for i, track_id in enumerate(track_ids):
        hue = int(180 * i / max(1, len(track_ids)))  # 0-180的色调值
        colors[track_id] = tuple(map(int, cv2.cvtColor(
            np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]))

    # 绘制每个追踪ID的轨迹和当前检测框
    for track_id, history in track_history.items():
        if not history:
            continue

        # 按帧数排序
        history.sort(key=lambda x: x['frame_count'])

        # 获取最近的历史点（最多history_points个）
        recent_history = history[-history_points:] if len(history) > history_points else history

        # 获取当前帧的信息（假设最后一个是最新的）
        current_info = history[-1]

        # 获取颜色
        color = colors[track_id]

        # 1. 绘制历史轨迹点
        for i, point_info in enumerate(recent_history):
            center_x, center_y = point_info['center']

            # 将浮点坐标转换为整数像素坐标
            pt = (int(center_x), int(center_y))

            # 根据点的远近调整大小和透明度（近大远小）
            alpha = 0.3 + 0.7 * (i / max(1, len(recent_history) - 1))  # 从0.3到1.0
            radius = max(1, int(3 * alpha))  # 半径从3到6

            # 创建带透明度的点
            overlay = img.copy()
            cv2.circle(overlay, pt, radius, color, -1)
            cv2.addWeighted(overlay, alpha * 0.5, img, 1 - alpha * 0.5, 0, img)

        # 2. 绘制当前帧的边界框
        x1, y1, x2, y2 = current_info['bbox']
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # 绘制边界框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # 3. 绘制当前中心点（比历史点大且更明显）
        current_center_x, current_center_y = current_info['center']
        current_center = (int(current_center_x), int(current_center_y))
        cv2.circle(img, current_center, 8, color, -1)
        cv2.circle(img, current_center, 8, (255, 255, 255), 2)  # 白色边框

        # 4. 绘制ID和类别标签
        label = f"ID:{track_id} {current_info['class_name']}"
        confidence = current_info['confidence']
        if confidence is not None:
            label += f" {confidence:.2f}"

        # 计算文本大小和位置
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2

        # 获取文本大小
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # 创建文本背景
        text_bg_top_left = (x1, max(0, y1 - text_height - 10))
        text_bg_bottom_right = (x1 + text_width + 10, y1)

        # 绘制半透明背景
        overlay = img.copy()
        cv2.rectangle(overlay, text_bg_top_left, text_bg_bottom_right, color, -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

        # 绘制文本
        text_pos = (x1 + 5, y1 - 5)
        cv2.putText(img, label, text_pos, font, font_scale, (255, 255, 255), thickness)

        # 5. 绘制从历史点到当前点的轨迹线（可选）
        if len(recent_history) > 1:
            points = [tuple(map(int, info['center'])) for info in recent_history]
            for i in range(1, len(points)):
                cv2.line(img, points[i - 1], points[i], color, 2)

    # 6. 在图像左上角添加统计信息
    stats_text = f"Vehicles: {len(track_ids)}"
    cv2.putText(img, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return img



def get_vehicle_class_ids(model):
    """获取车辆类别的ID"""
    vehicleIds = []
    vehicleClasses = ['car', 'bus', 'truck']
    for vehicle_class in vehicleClasses:
        # 查找类名对应的ID
        for idx, name in model.names.items():
            if vehicle_class in name.lower():
                vehicleIds.append(idx)
    print(f"Tracking vehicle classes: {vehicleClasses}")
    print(f"Corresponding class IDs: {vehicleIds}")
    return vehicleIds


# 加载DETRModel
def load_DETR_model(modelPath):
    DETRModel = YOLO(modelPath)
    vehicleClassIds = get_vehicle_class_ids(DETRModel)
    return DETRModel, vehicleClassIds


def vision_inference_thread(visualBasicInfoQueue, determinateBasicInfoQueue, imageQueue):
    videoDir = '/home/next_lb/桌面/next/DETR_BiLSTM/test_videos'
    videoNames = os.listdir(videoDir)
    DETRModelPath = '/home/next_lb/桌面/next/tempmodel/DETRX.pt'
    # 加载DETRModel
    DETRModel, vehicleClassIds = load_DETR_model(DETRModelPath)
    # 使用正则表达式提取数字部分
    videoNames= sorted(videoNames, key=lambda x: int(re.search(r'v(\d+)', x).group(1)))

    i = 0
    currentFrameCount = 0
    while True:
        if i >= len(videoNames):
            continue
        else:
            extractCarData = {}
            # 进行模型推理识别
            confThreshold = 0.2
            iouThreshold = 0.45
            videoPath = os.path.join(videoDir, videoNames[i])
            results = DETRModel.track(
                source=videoPath,
                conf=confThreshold,
                iou=iouThreshold,
                classes=vehicleClassIds,
                persist=True,
                tracker="botsort.yaml",
                # tracker="bytetrack.yaml",
                stream=True,
                verbose=False,
                half=False,
            )
            currentFrameCount = 0
            for result in results:
                currentFrameCount += 1
                imageQueue.put(result.orig_img.copy())
                imageHeight, imageWidth = result.orig_img.shape[:2]
                print(f'currentFrameCount: {currentFrameCount}')
                print(f"Window created with size: {imageWidth}x{imageHeight}")
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes.cpu().numpy()
                    if hasattr(boxes, 'id') and boxes.id is not None:
                        for j, box in enumerate(boxes):
                            # 获取类别信息
                            clsId = int(box.cls[0])
                            className = DETRModel.names[clsId]
                            confidence = float(box.conf[0])
                            # 获取边界框坐标
                            x1, y1, x2, y2 = box.xyxy[0]
                            # 获取追踪ID
                            trackId = int(box.id[0])
                            visualBasicInfoQueue.put((videoNames[i], className, trackId, float(x1), float(y1), float(x2), float(y2), confidence, currentFrameCount))
                            determinateBasicInfoQueue.put((videoNames[i], className, trackId, float(x1), float(y1), float(x2), float(y2), confidence, currentFrameCount, imageWidth, imageHeight))

                            # 记录到内存中，便于存入到数据集中
                            if f"{currentFrameCount}" not in extractCarData:
                                extractCarData[f"{currentFrameCount}"] = []
                            extractCarData[f"{currentFrameCount}"].append((className, trackId, float(x1), float(y1), float(x2), float(y2), confidence, currentFrameCount, imageWidth, imageHeight))

                time.sleep(0.1)



            # saveVehicleBehaviorDataPath = os.path.join(EXTRACT_BILSTM_RAW_DATA_PATH, f"{videoNames[i].split('.')[0]}_data.json")
            # # 保存为JSON文件
            # with open(saveVehicleBehaviorDataPath, 'w', encoding='utf-8') as f:
            #     json.dump(extractCarData, f, ensure_ascii=False, indent=4)


        i += 1
        time.sleep(1)




def visual_track_result_thread(visualBasicInfoQueue, imageQueue):
    nowVideoName = ''
    imageData = None
    basicInfo = None
    windowName = ''
    historyBasicInfo = []

    while True:

        while not imageQueue.empty():
            imageData = imageQueue.get()

        while not visualBasicInfoQueue.empty():
            basicInfo = visualBasicInfoQueue.get()
            historyBasicInfo.append(copy.deepcopy(basicInfo))

        if basicInfo:
            # 设定相关的图像显示的参数信息
            if nowVideoName == '':
                nowVideoName = basicInfo[0]
                windowName = f"{nowVideoName}: Visual Tracking Result"
                cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
                # 获取图像尺寸并设置窗口大小
                imageHeight, imageWidth = imageData.shape[:2]
                cv2.resizeWindow(windowName, imageWidth, imageHeight)

            else:
                # 如果更换视频了
                if nowVideoName != basicInfo[0]:
                    historyBasicInfo = []
                    # 循环结束后关闭所有OpenCV窗口
                    cv2.destroyAllWindows()
                    nowVideoName = basicInfo[0]
                    windowName = f"{nowVideoName}: Visual Tracking Result"
                    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
                    # 获取图像尺寸并设置窗口大小
                    imageHeight, imageWidth = imageData.shape[:2]
                    cv2.resizeWindow(windowName, imageWidth, imageHeight)


            # 绘制相关检测信息
            # 调用绘制函数
            resultImg = draw_detections_with_trajectories(
                image_data=imageData,
                history_basic_info=historyBasicInfo,
                history_points=20,
            )

            # 显示图像
            cv2.imshow(windowName, resultImg)

            # 等待1ms并检查按键
            # 如果按下'q'键则退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        basicInfo = None
        time.sleep(0.01)

    # 循环结束后关闭所有OpenCV窗口
    cv2.destroyAllWindows()




def determinate_vehicle_behavior_thread(determinateBasicInfoQueue):
    # 初始化轨迹行为判别对象
    BiLSTMTransformerInstance = BiLSTMTransformer()

    basicInfo = None
    while True:
        while not determinateBasicInfoQueue.empty():
            basicInfo = determinateBasicInfoQueue.get()
            # 将数据信息传入到判别算法实例中
            BiLSTMTransformerInstance.get_basic_info(basicInfo)

        # 进行行为的判定
        if basicInfo:
            # 数据预处理与准备
            BiLSTMTransformerInstance.data_preprocess_preparation()
            # 基于简单的规则的进行行为的判别
            BiLSTMTransformerInstance.conduct_behavior_assessment()
            # 使用机器学习等方式进行行为的判定
            BiLSTMTransformerInstance.determine_behavior_machine_learning()



        basicInfo = None
        time.sleep(0.01)


def main():


    # 这里启动三个线程，一个推理视频图像结果，一个实时可视化推理跟踪结果，一个实时判定检测车辆行为
    threadList = []
    imageQueue = Queue(maxsize=500)
    visualBasicInfoQueue = Queue(maxsize=500)
    determinateBasicInfoQueue = Queue(maxsize=500)
    visionInferenceTask = threading.Thread(target=vision_inference_thread, args=(visualBasicInfoQueue, determinateBasicInfoQueue, imageQueue, ), name="VisionInference")
    visualTrackTask = threading.Thread(target=visual_track_result_thread, args=(visualBasicInfoQueue, imageQueue, ), name="VisualTrack")
    determinateVehicleTask = threading.Thread(target=determinate_vehicle_behavior_thread, args=(determinateBasicInfoQueue, ), name="BehaviorDetermination")
    threadList.append(visionInferenceTask)
    threadList.append(visualTrackTask)
    threadList.append(determinateVehicleTask)
    # 启动线程队列
    for t in threadList:
        t.start()
        time.sleep(5)
    for t in threadList:
        t.join()
        print(f"Thread {t.name} finished")



if __name__ == '__main__':
    main()


