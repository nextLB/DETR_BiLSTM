import os
import cv2
import xml.etree.ElementTree as ET
import shutil
import yaml
import numpy as np
from pathlib import Path
import random
from tqdm import tqdm
import torch
from ultralytics import YOLO


def parse_detrac_xml(xml_path, img_width, img_height):
    """
    解析DETRAC XML文件，提取标注信息
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    annotations = {}

    # 获取序列中的所有帧
    for frame in root.findall('frame'):
        frame_num = int(frame.get('num'))
        target_list = frame.find('target_list')

        frame_annotations = []

        if target_list is not None:
            for target in target_list.findall('target'):
                target_id = target.get('id')
                box = target.find('box')

                if box is not None:
                    # 解析边界框坐标
                    left = float(box.get('left', 0))
                    top = float(box.get('top', 0))
                    width = float(box.get('width', 0))
                    height = float(box.get('height', 0))

                    # 计算边界框中心点和宽高（归一化）
                    x_center = (left + width / 2) / img_width
                    y_center = (top + height / 2) / img_height
                    w_norm = width / img_width
                    h_norm = height / img_height

                    # 确保坐标在[0, 1]范围内
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    w_norm = max(0, min(1, w_norm))
                    h_norm = max(0, min(1, h_norm))

                    # DETRAC数据集中车辆类型都是car，类别ID设为0
                    class_id = 0

                    frame_annotations.append({
                        'class_id': class_id,
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': w_norm,
                        'height': h_norm
                    })

        if frame_annotations:
            annotations[frame_num] = frame_annotations

    return annotations


def create_yolo_dataset_structure(base_dir, train_split=0.8):
    """
    创建YOLO格式的数据集结构
    """
    # 定义路径
    images_dir = Path("/home/next_lb/桌面/next/CAR_DETECTION_TRACK/data/train_data/DETRAC-Images/DETRAC-Images")
    train_ann_dir = Path(
        "/home/next_lb/桌面/next/CAR_DETECTION_TRACK/data/train_data/DETRAC-Train-Annotations-XML/DETRAC-Train-Annotations-XML")
    test_ann_dir = Path(
        "/home/next_lb/桌面/next/CAR_DETECTION_TRACK/data/train_data/DETRAC-Test-Annotations-XML/DETRAC-Test-Annotations-XML")

    # 创建输出目录
    output_dir = Path(base_dir) / "yolo_dataset"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建子目录
    (output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)

    # 处理训练集XML文件
    train_xml_files = list(train_ann_dir.glob("*.xml"))
    train_sequences = {}

    for xml_file in tqdm(train_xml_files, desc="处理训练集标注"):
        seq_name = xml_file.stem
        seq_img_dir = images_dir / seq_name

        if not seq_img_dir.exists():
            print(f"警告: 图像目录 {seq_img_dir} 不存在，跳过序列 {seq_name}")
            continue

        # 获取所有图像文件
        img_files = sorted(list(seq_img_dir.glob("*.jpg")))

        if not img_files:
            print(f"警告: 序列 {seq_name} 中没有图像文件")
            continue

        # 读取第一张图像获取尺寸
        first_img = cv2.imread(str(img_files[0]))
        if first_img is None:
            print(f"警告: 无法读取图像 {img_files[0]}")
            continue

        img_height, img_width = first_img.shape[:2]

        # 解析XML文件
        annotations = parse_detrac_xml(xml_file, img_width, img_height)

        train_sequences[seq_name] = {
            'img_dir': seq_img_dir,
            'annotations': annotations
        }

    # 处理测试集XML文件
    test_xml_files = list(test_ann_dir.glob("*.xml"))
    test_sequences = {}

    for xml_file in tqdm(test_xml_files, desc="处理测试集标注"):
        seq_name = xml_file.stem
        seq_img_dir = images_dir / seq_name

        if not seq_img_dir.exists():
            print(f"警告: 图像目录 {seq_img_dir} 不存在，跳过序列 {seq_name}")
            continue

        # 获取所有图像文件
        img_files = sorted(list(seq_img_dir.glob("*.jpg")))

        if not img_files:
            print(f"警告: 序列 {seq_name} 中没有图像文件")
            continue

        # 读取第一张图像获取尺寸
        first_img = cv2.imread(str(img_files[0]))
        if first_img is None:
            print(f"警告: 无法读取图像 {img_files[0]}")
            continue

        img_height, img_width = first_img.shape[:2]

        # 解析XML文件
        annotations = parse_detrac_xml(xml_file, img_width, img_height)

        test_sequences[seq_name] = {
            'img_dir': seq_img_dir,
            'annotations': annotations
        }

    # 准备训练/验证数据
    all_sequences = list(train_sequences.keys())
    random.shuffle(all_sequences)

    split_idx = int(len(all_sequences) * train_split)
    train_seqs = all_sequences[:split_idx]
    val_seqs = all_sequences[split_idx:]

    # 处理训练集序列
    for seq_name in tqdm(train_seqs, desc="准备训练数据"):
        seq_data = train_sequences[seq_name]
        seq_img_dir = seq_data['img_dir']
        annotations = seq_data['annotations']

        img_files = sorted(list(seq_img_dir.glob("*.jpg")))

        for img_file in img_files:
            # 提取帧号
            frame_num = int(img_file.stem.replace('img', ''))

            if frame_num in annotations:
                # 复制图像
                new_img_name = f"{seq_name}_{img_file.name}"
                dst_img_path = output_dir / "images" / "train" / new_img_name
                shutil.copy2(img_file, dst_img_path)

                # 创建标签文件
                label_content = []
                for ann in annotations[frame_num]:
                    line = f"{ann['class_id']} {ann['x_center']:.6f} {ann['y_center']:.6f} {ann['width']:.6f} {ann['height']:.6f}"
                    label_content.append(line)

                label_name = f"{seq_name}_{img_file.stem}.txt"
                label_path = output_dir / "labels" / "train" / label_name

                with open(label_path, 'w') as f:
                    f.write('\n'.join(label_content))

    # 处理验证集序列
    for seq_name in tqdm(val_seqs, desc="准备验证数据"):
        seq_data = train_sequences[seq_name]
        seq_img_dir = seq_data['img_dir']
        annotations = seq_data['annotations']

        img_files = sorted(list(seq_img_dir.glob("*.jpg")))

        for img_file in img_files:
            # 提取帧号
            frame_num = int(img_file.stem.replace('img', ''))

            if frame_num in annotations:
                # 复制图像
                new_img_name = f"{seq_name}_{img_file.name}"
                dst_img_path = output_dir / "images" / "val" / new_img_name
                shutil.copy2(img_file, dst_img_path)

                # 创建标签文件
                label_content = []
                for ann in annotations[frame_num]:
                    line = f"{ann['class_id']} {ann['x_center']:.6f} {ann['y_center']:.6f} {ann['width']:.6f} {ann['height']:.6f}"
                    label_content.append(line)

                label_name = f"{seq_name}_{img_file.stem}.txt"
                label_path = output_dir / "labels" / "val" / label_name

                with open(label_path, 'w') as f:
                    f.write('\n'.join(label_content))

    # 处理测试集（作为额外的验证集）
    for seq_name in tqdm(test_sequences.keys(), desc="准备测试数据"):
        seq_data = test_sequences[seq_name]
        seq_img_dir = seq_data['img_dir']
        annotations = seq_data['annotations']

        img_files = sorted(list(seq_img_dir.glob("*.jpg")))

        for img_file in img_files:
            # 提取帧号
            frame_num = int(img_file.stem.replace('img', ''))

            if frame_num in annotations:
                # 复制图像到验证集（因为测试集没有单独划分）
                new_img_name = f"{seq_name}_{img_file.name}"
                dst_img_path = output_dir / "images" / "val" / new_img_name
                shutil.copy2(img_file, dst_img_path)

                # 创建标签文件
                label_content = []
                for ann in annotations[frame_num]:
                    line = f"{ann['class_id']} {ann['x_center']:.6f} {ann['y_center']:.6f} {ann['width']:.6f} {ann['height']:.6f}"
                    label_content.append(line)

                label_name = f"{seq_name}_{img_file.stem}.txt"
                label_path = output_dir / "labels" / "val" / label_name

                with open(label_path, 'w') as f:
                    f.write('\n'.join(label_content))

    # 创建data.yaml配置文件
    yaml_content = {
        'path': str(output_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,  # 类别数量（car）
        'names': ['car'],  # 类别名称
        'download': None
    }

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print(f"\n数据集转换完成！")
    print(f"输出目录: {output_dir}")
    print(f"训练图像数量: {len(list((output_dir / 'images' / 'train').glob('*.jpg')))}")
    print(f"验证图像数量: {len(list((output_dir / 'images' / 'val').glob('*.jpg')))}")

    return str(output_dir), str(yaml_path)


def train_yolo_model(data_yaml_path, model_name='yolo11m.pt', epochs=100, imgsz=640, batch_size=16):
    """
    训练YOLO模型
    """
    print("\n" + "=" * 50)
    print("开始训练YOLOv11模型")
    print("=" * 50)

    # 检查GPU是否可用
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    print(f"GPU名称: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # 加载预训练模型
    print(f"加载模型: {model_name}")
    model = YOLO(model_name)

    # 训练参数配置
    train_args = {
        'data': data_yaml_path,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch_size,
        'device': device,
        'workers': 8,
        'patience': 50,  # 早停耐心值
        'save': True,
        'save_period': 10,  # 每10个epoch保存一次
        'cache': True,  # 缓存数据集以加速训练
        'project': 'runs/train',  # 保存训练结果的目录
        'name': 'yolov11_detrac_car_detection',
        'exist_ok': True,  # 允许覆盖现有项目
        'pretrained': True,  # 使用预训练权重
        'optimizer': 'auto',  # 自动选择优化器
        'verbose': True,  # 显示详细输出
        'seed': 42,  # 随机种子
        'deterministic': True,  # 确定性训练
        'single_cls': False,  # 多类别训练
        'rect': False,  # 矩形训练
        'cos_lr': True,  # 使用余弦学习率调度
        'label_smoothing': 0.1,  # 标签平滑
        'dropout': 0.0,  # dropout率
        'val': True,  # 在训练期间进行验证
        'amp': True,  # 自动混合精度训练
        'fraction': 1.0,  # 使用全部数据
        'profile': False,  # 不进行性能分析
        'overlap_mask': True,
        'mask_ratio': 4,
        'resume': False,  # 不从检查点恢复
    }

    # 开始训练
    print("\n开始训练...")
    results = model.train(**train_args)

    # 打印训练结果摘要
    print("\n" + "=" * 50)
    print("训练完成！")
    print("=" * 50)

    # 获取最佳模型路径
    best_model_path = Path('runs/train/yolov11_detrac_car_detection/weights/best.pt')
    if best_model_path.exists():
        print(f"最佳模型已保存到: {best_model_path}")

        # 加载最佳模型进行验证
        print("\n使用最佳模型进行验证...")
        best_model = YOLO(str(best_model_path))
        metrics = best_model.val(
            data=data_yaml_path,
            imgsz=imgsz,
            batch=batch_size,
            device=device,
            split='val'
        )

        # 打印评估指标
        print("\n验证结果:")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP75: {metrics.box.map75:.4f}")
        print(f"精确率: {metrics.box.p:.4f}")
        print(f"召回率: {metrics.box.r:.4f}")

        return str(best_model_path), metrics

    return None, None


def export_model(model_path, format='onnx'):
    """
    导出模型为不同格式
    """
    print(f"\n导出模型为{format.upper()}格式...")
    model = YOLO(model_path)

    if format == 'onnx':
        export_path = model.export(format='onnx', dynamic=True, simplify=True)
    elif format == 'torchscript':
        export_path = model.export(format='torchscript')
    elif format == 'engine':
        export_path = model.export(format='engine', half=True)
    else:
        export_path = model.export(format=format)

    print(f"模型已导出到: {export_path}")
    return export_path


def main():
    """
    主函数：转换数据集并训练模型
    """
    # 设置随机种子以保证可重复性
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    # 设置CUDA配置
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("=" * 60)
    print("DETRAC数据集转换和YOLOv11训练")
    print("=" * 60)

    # 步骤1: 创建YOLO格式数据集
    print("\n步骤1: 转换数据集为YOLO格式")
    dataset_dir = "/home/next_lb/桌面/next/CAR_DETECTION_TRACK"

    try:
        output_dir, data_yaml_path = create_yolo_dataset_structure(dataset_dir, train_split=0.8)
        print(f"数据集配置YAML文件: {data_yaml_path}")
    except Exception as e:
        print(f"数据集转换过程中出现错误: {e}")
        print("尝试使用现有的数据集配置...")
        # 如果转换失败，尝试使用现有的配置文件
        data_yaml_path = "/home/next_lb/桌面/next/CAR_DETECTION_TRACK/data/yolo_dataset/data.yaml"
        if not os.path.exists(data_yaml_path):
            print("未找到数据集配置文件，请检查路径！")
            return

    # 步骤2: 训练YOLO模型
    print("\n步骤2: 训练YOLOv11模型")

    # 训练参数
    model_name = 'yolo11m.pt'  # 使用yolov11m模型
    epochs = 100
    imgsz = 640
    batch_size = 16  # 根据GPU内存调整

    best_model_path, metrics = train_yolo_model(
        data_yaml_path=data_yaml_path,
        model_name=model_name,
        epochs=epochs,
        imgsz=imgsz,
        batch_size=batch_size
    )

    if best_model_path:
        # 步骤3: 导出模型（可选）
        print("\n步骤3: 导出模型")
        try:
            export_model(best_model_path, format='onnx')
        except Exception as e:
            print(f"模型导出失败: {e}")

    print("\n" + "=" * 60)
    print("训练流程完成！")
    print("=" * 60)

    # 提供使用模型的示例代码
    print("\n使用训练好的模型进行推理的示例代码:")
    print("```python")
    print("from ultralytics import YOLO")
    print("import cv2")
    print("")
    print("# 加载训练好的模型")
    print(
        f"model = YOLO('{best_model_path if best_model_path else 'runs/train/yolov11_detrac_car_detection/weights/best.pt'}')")
    print("")
    print("# 进行推理")
    print("results = model('path/to/your/image.jpg')")
    print("")
    print("# 可视化结果")
    print("for r in results:")
    print("    im_array = r.plot()  # 绘制边界框和标签")
    print("    cv2.imshow('Detection', im_array)")
    print("    cv2.waitKey(0)")
    print("    cv2.destroyAllWindows()")
    print("```")


if __name__ == "__main__":
    main()