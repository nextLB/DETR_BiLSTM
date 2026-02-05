


from ultralytics import YOLO
import os


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



def main():
    videoDir = '/home/next_lb/桌面/next/DETR_BiLSTM/test_videos'
    videoNames = os.listdir(videoDir)
    DETRModelPath = '/home/next_lb/桌面/next/tempmodel/DETRX.pt'

    # 加载DETRModel
    DETRModel, vehicleClassIds = load_DETR_model(DETRModelPath)
    print(DETRModel, vehicleClassIds)



if __name__ == '__main__':
    main()


