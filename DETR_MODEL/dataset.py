# dataset.py - 修正版
import os
import cv2
import torch
import numpy as np
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torchvision.transforms.functional as F
from PIL import Image

from config import Config


class DETRACDataset(Dataset):
    def __init__(self, images_root, annotations_path, split='train', transform=None):
        self.images_root = images_root
        self.annotations_path = annotations_path
        self.split = split
        self.transform = transform

        # 收集所有数据
        self.samples = self._load_samples()

        print(f"加载 {split} 数据集: {len(self.samples)} 个样本")

    def _load_samples(self):
        samples = []

        # 获取所有XML文件
        xml_files = [f for f in os.listdir(self.annotations_path) if f.endswith('.xml')]
        print(f"找到 {len(xml_files)} 个XML文件")

        for xml_file in xml_files[:50]:  # 先测试前50个，调试用
            xml_path = os.path.join(self.annotations_path, xml_file)
            sequence_name = xml_file.replace('.xml', '')
            sequence_dir = os.path.join(self.images_root, sequence_name)

            if not os.path.exists(sequence_dir):
                print(f"警告: 图像目录不存在: {sequence_dir}")
                continue

            # 解析XML
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
            except Exception as e:
                print(f"解析XML文件失败 {xml_path}: {e}")
                continue

            # 解析每一帧
            for frame_elem in root.findall('frame'):
                frame_num = int(frame_elem.get('num'))

                # 构建图像路径
                img_filename = f"img{frame_num:05d}.jpg"
                img_path = os.path.join(sequence_dir, img_filename)

                if not os.path.exists(img_path):
                    continue

                # 解析目标
                targets = []
                target_list = frame_elem.find('target_list')
                if target_list is not None:
                    for target_elem in target_list.findall('target'):
                        target_id = target_elem.get('id')
                        box_elem = target_elem.find('box')

                        if box_elem is not None:
                            # 获取边界框坐标
                            left = float(box_elem.get('left'))
                            top = float(box_elem.get('top'))
                            width = float(box_elem.get('width'))
                            height = float(box_elem.get('height'))

                            # 转换为[x_min, y_min, x_max, y_max]格式
                            x_min = left
                            y_min = top
                            x_max = left + width
                            y_max = top + height

                            # 车辆类别为1（0为背景）
                            targets.append({
                                'bbox': [x_min, y_min, x_max, y_max],
                                'category_id': 1  # 车辆
                            })

                # 添加样本（即使没有目标，用于负样本学习）
                samples.append({
                    'image_path': img_path,
                    'targets': targets,
                    'sequence': sequence_name,
                    'frame_num': frame_num
                })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 加载图像 - 使用PIL加载以确保兼容性
        try:
            image = Image.open(sample['image_path']).convert('RGB')
        except Exception as e:
            print(f"加载图像失败 {sample['image_path']}: {e}")
            # 返回一个空图像
            image = Image.new('RGB', (640, 480), (0, 0, 0))

        orig_w, orig_h = image.size

        # 获取目标
        targets = sample['targets']
        boxes = []
        labels = []

        for target in targets:
            bbox = target['bbox']
            # 归一化边界框坐标
            x_min = bbox[0] / orig_w
            y_min = bbox[1] / orig_h
            x_max = bbox[2] / orig_w
            y_max = bbox[3] / orig_h

            # 确保边界框有效且在[0,1]范围内
            x_min = max(0, min(1, x_min))
            y_min = max(0, min(1, y_min))
            x_max = max(0, min(1, x_max))
            y_max = max(0, min(1, y_max))

            # 确保边界框非空
            if x_max > x_min and y_max > y_min:
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(target['category_id'])

        # 如果没有目标，添加一个虚拟目标防止错误
        if len(boxes) == 0:
            boxes = [[0, 0, 0, 0]]
            labels = [0]  # 背景类

        # 应用transform（如果提供了）
        if self.transform:
            image = self.transform(image)
        else:
            # 默认transform
            transform = Compose([
                Resize(Config.IMG_SIZE),
                ToTensor(),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            image = transform(image)

        # 转换为tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # 创建目标字典（DETR格式）
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
        }

        return image, target


def collate_fn(batch):
    images = []
    targets = []

    for img, target in batch:
        images.append(img)
        targets.append(target)

    images = torch.stack(images, dim=0)
    return images, targets


def get_dataloaders():
    # 创建训练数据集
    print("加载训练数据集...")
    train_dataset = DETRACDataset(
        Config.IMAGES_ROOT,
        Config.TRAIN_ANNOTATIONS,
        split='train'
    )

    # 创建测试数据集
    print("加载验证数据集...")
    test_dataset = DETRACDataset(
        Config.IMAGES_ROOT,
        Config.TEST_ANNOTATIONS,
        split='test'
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=min(Config.NUM_WORKERS, 2),  # 减少workers以防内存问题
        collate_fn=collate_fn,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=min(Config.NUM_WORKERS, 2),
        collate_fn=collate_fn,
        pin_memory=True
    )

    return train_loader, test_loader