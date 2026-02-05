
import matplotlib as mpl
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
import threading
from queue import Queue
import time
import cv2
import sys
sys.path.append('/home/next_lb/桌面/next/DETR_BiLSTM')
from BiLSTM_TRS_MODEL.V2 import BiLSTMTransformer


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
    print(DETRModel, vehicleClassIds)

    i = 0
    while True:
        if i >= len(videoNames):
            continue
        else:
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
                half=False
            )
            for result in results:
                # 获取原始图像并转换为RGB（用于matplotlib）
                originalImage = cv2.cvtColor(result.orig_img.copy(), cv2.COLOR_BGR2RGB)
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
                            imageQueue.put(result.orig_img.copy(), originalImage)
                            visualBasicInfoQueue.put((videoNames[i], className, trackId, float(x1), float(y1), float(x2), float(y2), confidence))
                            determinateBasicInfoQueue.put((videoNames[i], className, trackId, float(x1), float(y1), float(x2), float(y2), confidence))
            i += 1
        time.sleep(0.1)



def visual_track_result_thread(visualBasicInfoQueue, imageQueue):
    # 设置matplotlib参数
    mpl.rcParams['toolbar'] = 'None'  # 隐藏工具栏
    # 初始化matplotlib
    plt.ion()  # 开启交互模式
    nowVideoName = ''
    imageData = None
    basicInfo = None
    while True:
        while not imageQueue.empty():
            imageData = imageQueue.get()
        while not visualBasicInfoQueue.empty():
            basicInfo = visualBasicInfoQueue.get()

        if imageData and basicInfo:
            # 设定相关的图像显示的参数信息
            if nowVideoName == '':
                nowVideoName = basicInfo[0]
                imageHeight, imageWidth = imageData[:2]
                print(imageWidth, imageWidth)
            else:
                print(nowVideoName)
                # 如果更换视频了
                if nowVideoName != basicInfo[0]:
                    pass

        imageData = None
        basicInfo = None
        time.sleep(1)


def determinate_vehicle_behavior_thread(determinateBasicInfoQueue):
    while True:
        time.sleep(1)


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


