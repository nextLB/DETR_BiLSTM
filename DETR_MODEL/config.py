# config.py
import torch


class Config:
    # 数据集路径
    IMAGES_ROOT = "/home/next_lb/桌面/next/CAR_DETECTION_TRACK/data/train_data/DETRAC-Images/DETRAC-Images/"
    TRAIN_ANNOTATIONS = "/home/next_lb/桌面/next/CAR_DETECTION_TRACK/data/train_data/DETRAC-Train-Annotations-XML/DETRAC-Train-Annotations-XML/"
    TEST_ANNOTATIONS = "/home/next_lb/桌面/next/CAR_DETECTION_TRACK/data/train_data/DETRAC-Test-Annotations-XML/DETRAC-Test-Annotations-XML/"

    # 训练参数
    BATCH_SIZE = 4
    NUM_WORKERS = 4
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4

    # 模型参数
    NUM_CLASSES = 2  # 背景 + 车辆
    HIDDEN_DIM = 256
    NHEADS = 8
    NUM_ENCODER_LAYERS = 6
    NUM_DECODER_LAYERS = 6
    DROPOUT = 0.1
    DIM_FEEDFORWARD = 2048

    # 图像参数
    IMG_SIZE = (800, 1333)  # DETR标准输入尺寸

    # 设备
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 保存路径
    OUTPUT_DIR = "/home/next_lb/桌面/next/CAR_DETECTION_TRACK/outputs/"
    MODEL_SAVE_PATH = OUTPUT_DIR + "detr_model.pth"

    # DETR参数
    NUM_QUERIES = 100
    AUX_LOSS = True


