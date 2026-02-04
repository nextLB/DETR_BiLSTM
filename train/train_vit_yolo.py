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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
import math
from collections import OrderedDict
import time
from datetime import datetime


# ==================== 自主构建的 Vision Transformer 模块 ====================
class PatchEmbedding(nn.Module):
    """将图像分割成补丁并嵌入"""

    def __init__(self, img_size=640, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # 使用卷积实现补丁嵌入（更高效）
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

        # 层归一化
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, embed_dim, H/patch_size, W/patch_size]
        x = x.flatten(2)  # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, embed_dim]
        x = self.norm(x)
        return x


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, seq_len, d_model]
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""

    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim必须能被num_heads整除"

        # Q, K, V 投影
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # 输出投影
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        # 缩放因子
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        B, N, C = x.shape

        # 线性投影
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # 应用注意力权重
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # 输出投影
        x = self.out_proj(x)
        x = self.proj_dropout(x)

        return x


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""

    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        # 自注意力
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        # MLP
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 残差连接 + 层归一化
        x = x + self.dropout(self.self_attn(self.norm1(x)))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    """自主实现的Vision Transformer"""

    def __init__(self, img_size=640, patch_size=16, in_channels=3,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
                 dropout=0.1, num_classes=0):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # 补丁嵌入
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)

        # 类别token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 位置编码
        self.pos_embed = PositionalEncoding(embed_dim, max_len=self.num_patches + 1)

        # Transformer编码器
        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # 层归一化
        self.norm = nn.LayerNorm(embed_dim)

        # 分类头（如果使用）
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        # 初始化类别token
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # 初始化线性层和层归一化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, return_features=False):
        # 补丁嵌入
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]

        # 添加类别token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # 添加位置编码
        x = self.pos_embed(x)

        # 通过Transformer块
        intermediate_features = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            # 保存中间层特征
            if i in [len(self.blocks) // 4, len(self.blocks) // 2, 3 * len(self.blocks) // 4]:
                intermediate_features.append(x)

        # 添加最后一层特征
        intermediate_features.append(x)

        # 层归一化
        x = self.norm(x)

        if return_features:
            return x, intermediate_features
        return x


# ==================== YOLOv11 检测头部模块 ====================
class Conv(nn.Module):
    """标准卷积层"""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, groups=1, dilation=1,
                 activation=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding=autopad(kernel_size, padding, dilation),
                              groups=groups, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if activation else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


def autopad(kernel_size, padding=None, dilation=1):
    """自动计算填充大小"""
    if dilation > 1:
        kernel_size = dilation * (kernel_size - 1) + 1
    if padding is None:
        padding = kernel_size // 2
    return padding


class Bottleneck(nn.Module):
    """标准瓶颈层"""

    def __init__(self, in_channels, out_channels, shortcut=True, groups=1, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.cv1 = Conv(in_channels, hidden_channels, 1, 1)
        self.cv2 = Conv(hidden_channels, out_channels, 3, 1, groups=groups)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """C2f模块，来自YOLOv11"""

    def __init__(self, in_channels, out_channels, n=1, shortcut=False, groups=1, expansion=0.5):
        super().__init__()
        self.c = int(out_channels * expansion)  # 隐藏通道数
        self.cv1 = Conv(in_channels, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, out_channels, 1)

        # 构建瓶颈层
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, groups, expansion=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """空间金字塔池化快速版"""

    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        hidden_channels = in_channels // 2
        self.cv1 = Conv(in_channels, hidden_channels, 1, 1)
        self.cv2 = Conv(hidden_channels * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1,
                              padding=kernel_size // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))


class Detect(nn.Module):
    """YOLO检测头"""

    def __init__(self, num_classes=80, channels=()):
        super().__init__()
        self.num_classes = num_classes
        self.num_outputs = num_classes + 5  # xywh + obj_conf + cls_conf
        self.num_layers = len(channels)

        # 为每个检测层创建检测头
        self.m = nn.ModuleList()
        for in_channels in channels:
            self.m.append(
                nn.Conv2d(in_channels, self.num_outputs, 1)
            )

    def forward(self, x):
        return [m(level) for m, level in zip(self.m, x)]


# ==================== ViT-YOLO 完整模型 ====================
class ViTYOLOModel(nn.Module):
    """ViT + YOLOv11 完整模型"""

    def __init__(self, num_classes=1, img_size=640, patch_size=16,
                 embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes

        # Vision Transformer 骨干网络
        self.vit = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=4.0,
            dropout=0.1
        )

        # 计算ViT输出特征图的分辨率
        self.patch_size = patch_size
        self.feat_size = img_size // patch_size

        # 特征调整层（将ViT输出转换为2D特征图）
        self.feat_adjust = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Linear(embed_dim // 2, embed_dim // 4),
                nn.LayerNorm(embed_dim // 4)
            ) for _ in range(3)
        ])

        # 通道调整层 - 输出固定通道数
        self.channel_adjust = nn.ModuleList([
            Conv(embed_dim // 4, 128, 1),  # 浅层特征：128通道
            Conv(embed_dim // 4, 256, 1),  # 中层特征：256通道
            Conv(embed_dim // 4, 512, 1)  # 深层特征：512通道
        ])

        # 特征融合时需要的卷积层（预定义）
        self.deep_to_mid_conv = Conv(512, 256, 1)  # 深层到中层的通道调整
        self.mid_to_shallow_conv = Conv(256, 128, 1)  # 中层到浅层的通道调整

        # 特征融合后的通道调整层
        self.fuse_mid_conv = Conv(512, 256, 1)  # 中层特征融合后调整（512 -> 256）
        self.fuse_shallow_conv = Conv(256, 128, 1)  # 浅层特征融合后调整（256 -> 128）

        # FPN结构
        self.upsample2x = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample4x = nn.Upsample(scale_factor=4, mode='nearest')
        self.downsample2x = nn.MaxPool2d(kernel_size=2, stride=2)

        # 特征融合层
        self.fusion_layers = nn.ModuleList([
            C2f(128, 128, n=1, shortcut=False),  # 浅层
            C2f(256, 256, n=1, shortcut=False),  # 中层
            C2f(512, 512, n=1, shortcut=False)  # 深层
        ])

        # SPPF层
        self.sppf = SPPF(512, 512, kernel_size=5)

        # 检测头
        self.detect = Detect(num_classes, [128, 256, 512])

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        # 初始化所有层
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # 通过ViT骨干网络，获取特征
        vit_output, intermediate_features = self.vit(x, return_features=True)

        # 确保我们有足够的中间特征
        if len(intermediate_features) < 3:
            while len(intermediate_features) < 3:
                intermediate_features.append(vit_output)

        # 选择三个不同深度的特征
        shallow_idx = min(1, len(intermediate_features) - 1)
        mid_idx = min(len(intermediate_features) // 2, len(intermediate_features) - 1)
        deep_idx = -1

        selected_features = [
            intermediate_features[shallow_idx],
            intermediate_features[mid_idx],
            intermediate_features[deep_idx]
        ]

        # 处理每个特征
        spatial_features = []
        for i, feat in enumerate(selected_features):
            B = feat.shape[0]

            # 检查特征形状并移除类别token（如果存在）
            if feat.dim() == 3:
                if feat.shape[1] == self.feat_size * self.feat_size + 1:
                    feat = feat[:, 1:, :]
                elif feat.shape[1] != self.feat_size * self.feat_size:
                    feat = feat[:, :self.feat_size * self.feat_size, :]
            else:
                # 添加一个零特征图作为占位符
                dummy_feat = torch.zeros(B, self.feat_size * self.feat_size,
                                         self.embed_dim // 4, device=x.device)
                spatial_features.append(dummy_feat)
                continue

            # 调整特征维度
            feat = self.feat_adjust[i](feat)  # [B, num_patches, embed_dim//4]

            # 重塑为2D特征图
            B, N, C = feat.shape

            # 确保我们能够重塑
            if N != self.feat_size * self.feat_size:
                # 插值到正确大小
                feat = feat.reshape(B, self.feat_size, self.feat_size, C).permute(0, 3, 1, 2)
            else:
                H = W = int(math.sqrt(N))
                feat = feat.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]

            # 调整通道数
            feat = self.channel_adjust[i](feat)

            # 将所有特征图调整到相同的空间尺寸
            target_size = self.img_size // 16
            if feat.shape[2] != target_size or feat.shape[3] != target_size:
                feat = F.interpolate(feat, size=(target_size, target_size),
                                     mode='bilinear', align_corners=False)

            spatial_features.append(feat)

        # 确保我们有3个特征图
        while len(spatial_features) < 3:
            # 添加一个零特征图
            if spatial_features:
                dummy_feat = torch.zeros_like(spatial_features[0])
            else:
                dummy_feat = torch.zeros(B, 128, target_size, target_size, device=x.device)
            spatial_features.append(dummy_feat)

        # 特征金字塔融合
        x_deep = spatial_features[2]  # [B, 512, H, W]
        x_deep = self.sppf(x_deep)
        x_deep = self.fusion_layers[2](x_deep)

        # 中层特征处理
        x_mid = spatial_features[1]  # [B, 256, H, W]
        x_mid = self.fusion_layers[1](x_mid)

        # 浅层特征处理
        x_shallow = spatial_features[0]  # [B, 128, H, W]
        x_shallow = self.fusion_layers[0](x_shallow)

        # 创建输出列表（注意：YOLO检测头通常期望浅层到深层的顺序）
        # 但根据错误信息，我们需要调整顺序
        # 让我们先打印一下通道数来调试
        if not hasattr(self, 'debug_printed'):
            print(f"特征图通道数: 浅层={x_shallow.shape[1]}, 中层={x_mid.shape[1]}, 深层={x_deep.shape[1]}")
            self.debug_printed = True

        # 根据检测头的期望重新排序
        outputs = []
        for feat in [x_shallow, x_mid, x_deep]:
            # 确保特征图通道数与检测头匹配
            if feat.shape[1] not in [128, 256, 512]:
                # 调整通道数
                if feat.shape[1] < 256:
                    feat = Conv(feat.shape[1], 128, 1).to(x.device)(feat)
                elif feat.shape[1] < 512:
                    feat = Conv(feat.shape[1], 256, 1).to(x.device)(feat)
                else:
                    feat = Conv(feat.shape[1], 512, 1).to(x.device)(feat)
            outputs.append(feat)

        # 检测
        return self.detect(outputs)

# ==================== 数据集和数据加载器 ====================
class DETRACDataset(Dataset):
    """DETRAC数据集类"""

    def __init__(self, images_dir, labels_dir, img_size=640, augment=False):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.img_size = img_size
        self.augment = augment

        # 获取所有图像文件
        self.image_files = sorted(list(self.images_dir.glob("*.jpg")))

        # 检查对应的标签文件是否存在
        self.valid_files = []
        for img_path in self.image_files:
            label_path = self.labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                self.valid_files.append(img_path)

        print(f"找到 {len(self.valid_files)} 个有效图像-标签对")

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        img_path = self.valid_files[idx]
        label_path = self.labels_dir / f"{img_path.stem}.txt"

        # 读取图像
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"无法读取图像: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        # 读取标签
        bboxes = []
        class_labels = []

        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:])

                    # 转换到像素坐标
                    x1 = (x_center - width / 2) * orig_w
                    y1 = (y_center - height / 2) * orig_h
                    x2 = (x_center + width / 2) * orig_w
                    y2 = (y_center + height / 2) * orig_h

                    # 确保坐标在图像范围内
                    x1 = max(0, min(orig_w - 1, x1))
                    y1 = max(0, min(orig_h - 1, y1))
                    x2 = max(0, min(orig_w - 1, x2))
                    y2 = max(0, min(orig_h - 1, y2))

                    if x2 > x1 and y2 > y1:  # 确保边界框有效
                        bboxes.append([x1, y1, x2, y2])
                        class_labels.append(class_id)

        # 数据增强
        if self.augment and len(bboxes) > 0:
            # 随机水平翻转
            if random.random() > 0.5:
                image = cv2.flip(image, 1)
                for i, bbox in enumerate(bboxes):
                    x1, y1, x2, y2 = bbox
                    new_x1 = orig_w - x2
                    new_x2 = orig_w - x1
                    bboxes[i] = [new_x1, y1, new_x2, y2]

            # 随机亮度调整
            brightness_factor = random.uniform(0.8, 1.2)
            image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)

        # 调整图像大小
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = image.astype(np.float32) / 255.0

        # 调整边界框坐标
        scale_x = self.img_size / orig_w
        scale_y = self.img_size / orig_h
        normalized_bboxes = []

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            # 转换到新尺寸
            x1 = x1 * scale_x / self.img_size
            y1 = y1 * scale_y / self.img_size
            x2 = x2 * scale_x / self.img_size
            y2 = y2 * scale_y / self.img_size

            # 计算中心点和宽高
            width = max(0.01, x2 - x1)
            height = max(0.01, y2 - y1)
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2

            normalized_bboxes.append([x_center, y_center, width, height])

        # 转换为张量
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()

        # 准备标签
        labels = []
        for cls, bbox in zip(class_labels, normalized_bboxes):
            labels.append([cls] + bbox)

        if len(labels) == 0:
            labels_tensor = torch.zeros((0, 5))
        else:
            labels_tensor = torch.tensor(labels)

        return image_tensor, labels_tensor

    @staticmethod
    def collate_fn(batch):
        """批处理函数"""
        images, labels = zip(*batch)
        images = torch.stack(images, 0)

        # 创建一个列表来存储每个图像的标签
        labels_list = []
        for label in labels:
            labels_list.append(label)

        return images, labels_list


# ==================== 损失函数 ====================
class YOLOLoss(nn.Module):
    """YOLO损失函数 - 简化版本"""

    def __init__(self, num_classes=1, anchors=None):
        super().__init__()
        self.num_classes = num_classes

        # 损失权重
        self.box_weight = 0.05
        self.obj_weight = 1.0
        self.cls_weight = 0.5

        # 如果没有提供anchors，使用默认值
        if anchors is None:
            # 根据特征图大小调整的anchors
            self.anchors = [
                [(0.028, 0.038), (0.038, 0.048), (0.048, 0.058)],  # 大特征图 (80x80)
                [(0.068, 0.088), (0.088, 0.108), (0.108, 0.128)],  # 中特征图 (40x40)
                [(0.148, 0.188), (0.188, 0.228), (0.228, 0.268)]  # 小特征图 (20x20)
            ]
        else:
            self.anchors = anchors

        # 损失函数
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, predictions, targets):
        """
        简化的YOLO损失计算
        predictions: 模型输出，三个尺度的特征图
        targets: 目标边界框列表
        """
        total_loss = 0
        device = predictions[0].device

        # 处理每个尺度的预测
        for scale_idx, pred in enumerate(predictions):
            if pred is None:
                continue

            B, C, H, W = pred.shape

            # 重塑预测张量
            pred = pred.view(B, C, H, W).permute(0, 2, 3, 1).contiguous()
            pred = pred.view(B, H, W, -1)  # [B, H, W, 5+num_classes]

            # 如果没有目标，只计算背景损失
            if all(len(t) == 0 for t in targets):
                # 创建全零的目标置信度
                obj_target = torch.zeros(B, H, W, 1, device=device)
                obj_pred = pred[..., 4:5]  # 目标置信度预测
                obj_loss = F.binary_cross_entropy_with_logits(obj_pred, obj_target)
                total_loss += self.obj_weight * obj_loss
                continue

            # 初始化目标张量
            box_target = torch.zeros(B, H, W, 4, device=device)
            obj_target = torch.zeros(B, H, W, 1, device=device)
            cls_target = torch.zeros(B, H, W, self.num_classes, device=device)

            # 为每个样本分配目标
            for b in range(B):
                target = targets[b]
                if len(target) == 0:
                    continue

                # 转换目标框
                target_boxes = target[:, 1:5]  # [N, 4]
                target_cls = target[:, 0].long()  # [N]

                # 将目标框分配到网格单元
                for i in range(len(target)):
                    x_center, y_center, width, height = target_boxes[i]

                    # 计算网格索引
                    grid_x = int(x_center * W)
                    grid_y = int(y_center * H)

                    # 确保索引在范围内
                    grid_x = min(max(grid_x, 0), W - 1)
                    grid_y = min(max(grid_y, 0), H - 1)

                    # 设置目标值
                    box_target[b, grid_y, grid_x] = torch.tensor([x_center, y_center, width, height])
                    obj_target[b, grid_y, grid_x] = 1.0

                    # 如果是多类别，设置类别目标
                    if self.num_classes > 1:
                        cls_target[b, grid_y, grid_x, target_cls[i]] = 1.0

            # 提取预测值
            pred_boxes = pred[..., :4]
            pred_obj = pred[..., 4:5]
            pred_cls = pred[..., 5:]

            # 计算边界框损失 (MSE)
            box_loss = F.mse_loss(pred_boxes.sigmoid(), box_target, reduction='mean')

            # 计算目标置信度损失
            obj_loss = F.binary_cross_entropy_with_logits(pred_obj, obj_target)

            # 计算类别损失
            if self.num_classes > 1:
                cls_loss = F.binary_cross_entropy_with_logits(pred_cls, cls_target)
            else:
                cls_loss = torch.tensor(0.0, device=device)

            # 总损失
            scale_loss = (
                    self.box_weight * box_loss +
                    self.obj_weight * obj_loss +
                    self.cls_weight * cls_loss
            )

            total_loss += scale_loss

        # 如果所有预测都是None，返回一个小损失
        if total_loss == 0:
            return torch.tensor(0.1, device=device, requires_grad=True)

        return total_loss / len(predictions)


# ==================== 训练函数 ====================
def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)

        # 前向传播
        optimizer.zero_grad()
        predictions = model(images)

        # 计算损失
        loss = criterion(predictions, targets)

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 更新进度条
        pbar.set_postfix({'loss': loss.item()})

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc='Validating'):
            images = images.to(device)

            # 前向传播
            predictions = model(images)

            # 计算损失
            loss = criterion(predictions, targets)
            total_loss += loss.item()

    return total_loss / len(dataloader)


# ==================== 主训练函数 ====================
def train_vit_yolo_model(data_yaml_path, epochs=100, batch_size=16, img_size=640):
    """训练ViT-YOLO模型"""
    print("\n" + "=" * 50)
    print("开始训练 ViT-YOLO 模型（完全自主实现）")
    print("=" * 50)

    # 加载数据配置
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")

    # 创建模型
    print("创建 ViT-YOLO 模型...")
    model = ViTYOLOModel(
        num_classes=1,  # 只有car类别
        img_size=img_size,
        patch_size=16,
        embed_dim=512,  # 减小维度以节省内存
        depth=8,  # 减小深度以加快训练
        num_heads=8
    )
    model = model.to(device)

    # 输出模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 创建数据集
    print("创建数据集...")
    dataset_base = Path(data_config['path'])
    train_dataset = DETRACDataset(
        images_dir=dataset_base / data_config['train'],
        labels_dir=dataset_base / 'labels/train',
        img_size=img_size,
        augment=True
    )

    val_dataset = DETRACDataset(
        images_dir=dataset_base / data_config['val'],
        labels_dir=dataset_base / 'labels/val',
        img_size=img_size,
        augment=False
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=DETRACDataset.collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=DETRACDataset.collate_fn
    )

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")

    # 损失函数
    criterion = YOLOLoss(num_classes=1)

    # 优化器
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # 学习率调度器
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # 训练循环
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    # 创建保存目录
    save_dir = Path('vit_yolo_checkpoints')
    save_dir.mkdir(exist_ok=True)

    for epoch in range(epochs):
        print(f"\n{'=' * 40}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'=' * 40}")

        # 训练
        train_loss = train_epoch(model, train_loader, optimizer,
                                 criterion, device, epoch + 1)
        train_losses.append(train_loss)

        # 验证
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        print(f"训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': {
                    'num_classes': 1,
                    'img_size': img_size,
                    'patch_size': 16,
                    'embed_dim': 512,
                    'depth': 8,
                    'num_heads': 8
                }
            }
            torch.save(checkpoint, save_dir / 'best_vit_yolo.pth')
            print(f"保存最佳模型，验证损失: {val_loss:.4f}")

        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }
            torch.save(checkpoint, save_dir / f'checkpoint_epoch_{epoch + 1}.pth')
            print(f"保存检查点: epoch {epoch + 1}")

        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"当前学习率: {current_lr:.6f}")

    print(f"\n训练完成！最佳验证损失: {best_val_loss:.4f}")

    # 保存最终模型
    final_model_path = save_dir / 'vit_yolo_final.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f"最终模型已保存到: {final_model_path}")

    # 绘制损失曲线
    plot_loss_curve(train_losses, val_losses, save_dir)

    return model, best_val_loss


def plot_loss_curve(train_losses, val_losses, save_dir):
    """绘制损失曲线"""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失', linewidth=2)
    plt.plot(val_losses, label='验证损失', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title('训练和验证损失曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_dir / 'loss_curve.png', dpi=300, bbox_inches='tight')
    plt.close()


# ==================== 推理函数 ====================
def predict(model, image_path, device, img_size=640, conf_threshold=0.5):
    """使用训练好的模型进行预测"""
    # 加载图像
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")

    orig_h, orig_w = image.shape[:2]

    # 预处理
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (img_size, img_size))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # 推理
    model.eval()
    with torch.no_grad():
        predictions = model(img_tensor)

    # 解析预测结果
    detections = []

    # 不同尺度的锚框尺寸（根据经验设置）
    anchor_sizes = [
        [(0.028, 0.038), (0.038, 0.048), (0.048, 0.058)],  # 小目标
        [(0.068, 0.088), (0.088, 0.108), (0.108, 0.128)],  # 中目标
        [(0.148, 0.188), (0.188, 0.228), (0.228, 0.268)]  # 大目标
    ]

    for scale_idx, pred in enumerate(predictions):
        if pred is None:
            continue

        B, C, H, W = pred.shape

        # 解析预测
        pred = pred.view(B, 6, H, W)  # 4+1+1=6 (xywh + obj_conf + cls_conf)
        pred = pred.permute(0, 2, 3, 1).contiguous()

        # 创建网格
        grid_x, grid_y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
        grid = torch.stack([grid_x, grid_y], dim=-1).float().to(device)

        # 提取预测值
        pred_bbox = torch.sigmoid(pred[..., :4])
        pred_obj = torch.sigmoid(pred[..., 4])

        # 转换边界框坐标
        for i in range(H):
            for j in range(W):
                if pred_obj[0, i, j] > conf_threshold:
                    # 边界框参数
                    dx, dy, dw, dh = pred_bbox[0, i, j].cpu().numpy()

                    # 转换为绝对坐标
                    x_center = (j + dx) / W
                    y_center = (i + dy) / H
                    width = dw
                    height = dh

                    # 置信度
                    confidence = pred_obj[0, i, j].item()

                    # 转换到原始图像坐标
                    x1 = int((x_center - width / 2) * orig_w)
                    y1 = int((y_center - height / 2) * orig_h)
                    x2 = int((x_center + width / 2) * orig_w)
                    y2 = int((y_center + height / 2) * orig_h)

                    # 确保坐标在图像范围内
                    x1 = max(0, min(orig_w - 1, x1))
                    y1 = max(0, min(orig_h - 1, y1))
                    x2 = max(0, min(orig_w - 1, x2))
                    y2 = max(0, min(orig_h - 1, y2))

                    if x2 > x1 and y2 > y1:  # 确保边界框有效
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': confidence,
                            'class': 0,  # car类别
                            'class_name': 'car'
                        })

    return detections


def visualize_predictions(image_path, detections, output_path=None, conf_threshold=0.3):
    """可视化预测结果"""
    image = cv2.imread(str(image_path))

    # 按置信度排序
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

    # 非极大值抑制
    if len(detections) > 0:
        boxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['confidence'] for d in detections])

        # 简单的非极大值抑制
        indices = []
        while len(boxes) > 0:
            # 选择最高置信度的框
            best_idx = np.argmax(scores)
            indices.append(best_idx)
            best_box = boxes[best_idx]

            # 计算IoU
            ious = []
            for box in boxes:
                iou = compute_iou(best_box, box)
                ious.append(iou)

            ious = np.array(ious)

            # 移除重叠度高的框
            keep_mask = ious < 0.5
            boxes = boxes[keep_mask]
            scores = scores[keep_mask]

        # 只保留筛选后的检测结果
        detections = [detections[i] for i in indices]

    for det in detections:
        if det['confidence'] < conf_threshold:
            continue

        x1, y1, x2, y2 = det['bbox']
        conf = det['confidence']
        class_name = det['class_name']

        # 绘制边界框
        color = (0, 255, 0)  # 绿色
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # 绘制标签背景
        label = f"{class_name}: {conf:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

        cv2.rectangle(image,
                      (x1, y1 - text_height - baseline - 5),
                      (x1 + text_width, y1),
                      color, -1)

        # 绘制标签文本
        cv2.putText(image, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    if output_path:
        cv2.imwrite(str(output_path), image)

    return image


def compute_iou(box1, box2):
    """计算两个边界框的IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    return inter_area / (union_area + 1e-7)


# ==================== 主函数 ====================
def main():
    """主函数"""
    # 设置随机种子
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
    print("ViT-YOLO 目标检测模型训练（完全自主实现）")
    print("=" * 60)

    # 数据集路径
    dataset_dir = "/home/next_lb/桌面/next/CAR_DETECTION_TRACK"
    data_yaml_path = f"{dataset_dir}/yolo_dataset/data.yaml"

    # 检查配置文件是否存在
    if not os.path.exists(data_yaml_path):
        print(f"未找到数据集配置文件: {data_yaml_path}")
        print("请先运行数据集转换代码！")
        return

    # 训练参数
    epochs = 50  # 可根据需要调整
    batch_size = 8  # 根据GPU内存调整
    img_size = 480

    # 步骤1: 训练模型
    print("\n步骤1: 训练ViT-YOLO模型")
    try:
        model, best_val_loss = train_vit_yolo_model(
            data_yaml_path=data_yaml_path,
            epochs=epochs,
            batch_size=batch_size,
            img_size=img_size
        )
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return

    # 步骤2: 测试模型
    print("\n步骤2: 测试模型")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载最佳模型
    best_model_path = Path('vit_yolo_checkpoints/best_vit_yolo.pth')
    if best_model_path.exists():
        print(f"加载最佳模型: {best_model_path}")

        # 创建模型实例
        # 在train_vit_yolo_model函数中修改模型创建
        model = ViTYOLOModel(
            num_classes=1,  # 只有car类别
            img_size=img_size,
            patch_size=16,
            embed_dim=384,  # 减小嵌入维度
            depth=6,  # 减少Transformer层数
            num_heads=6  # 减少注意力头数
        )
        model = model.to(device)

        # 加载权重
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        print("模型加载成功！")

        # 寻找测试图像
        test_images = []
        test_dir = Path(dataset_dir) / "data/yolo_dataset/images/val"
        if test_dir.exists():
            test_images = list(test_dir.glob("*.jpg"))

        if test_images:
            # 测试前5张图像
            for i in range(min(5, len(test_images))):
                test_image_path = test_images[i]
                print(f"\n测试图像 {i + 1}: {test_image_path.name}")

                try:
                    # 预测
                    detections = predict(model, test_image_path, device)

                    # 可视化
                    output_path = f"test_result_{i + 1}.jpg"
                    result_image = visualize_predictions(
                        test_image_path,
                        detections,
                        output_path,
                        conf_threshold=0.3
                    )

                    print(f"检测到 {len(detections)} 辆车")
                    print(f"结果已保存到: {output_path}")

                except Exception as e:
                    print(f"处理图像时出错: {e}")
        else:
            print("未找到测试图像！")

    print("\n" + "=" * 60)
    print("训练流程完成！")
    print("=" * 60)

    # 提供使用模型的示例代码
    print("\n使用训练好的模型进行推理的示例代码:")
    print("```python")
    print("import torch")
    print("import cv2")
    print("from model import ViTYOLOModel, predict, visualize_predictions")
    print("")
    print("# 加载模型")
    print("device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')")
    print("model = ViTYOLOModel(num_classes=1, img_size=640)")
    print("model.load_state_dict(torch.load('vit_yolo_checkpoints/best_vit_yolo.pth')['model_state_dict'])")
    print("model = model.to(device)")
    print("model.eval()")
    print("")
    print("# 进行推理")
    print("image_path = 'path/to/your/image.jpg'")
    print("detections = predict(model, image_path, device)")
    print("")
    print("# 可视化结果")
    print("result = visualize_predictions(image_path, detections, 'output.jpg')")
    print("cv2.imshow('Detection', result)")
    print("cv2.waitKey(0)")
    print("cv2.destroyAllWindows()")
    print("```")


if __name__ == "__main__":
    # 检查必要的库
    try:
        import cv2
        import torch
        import yaml
        from tqdm import tqdm
    except ImportError as e:
        print(f"缺少必要的库: {e}")
        print("请运行: pip install opencv-python torch torchvision PyYAML tqdm")
        exit(1)

    # 运行主函数
    main()