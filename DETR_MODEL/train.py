# train.py - 修改版（无WandB）
import os
import time
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')
from scipy.optimize import linear_sum_assignment

from config import Config
from dataset import get_dataloaders
from model import DETR, PostProcess
from loss import build_criterion


def save_checkpoint(model, optimizer, scheduler, epoch, loss, path, is_best=False):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
    }

    torch.save(checkpoint, path)

    if is_best:
        best_path = path.replace('.pth', '_best.pth')
        torch.save(checkpoint, best_path)
        print(f"保存最佳模型到: {best_path}")


def load_checkpoint(model, optimizer, scheduler, path):
    """加载检查点"""
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=Config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and checkpoint['optimizer_state_dict']:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['loss']
        print(f"加载检查点: epoch {checkpoint['epoch']}, loss {best_loss:.4f}")
        return start_epoch, best_loss
    return 0, float('inf')


def train_epoch(model, dataloader, criterion, optimizer, scaler, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    losses_dict = {}

    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')

    for batch_idx, (images, targets) in enumerate(progress_bar):
        images = images.to(Config.DEVICE)
        targets = [{k: v.to(Config.DEVICE) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        # 混合精度训练
        with autocast():
            outputs = model(images)
            losses = criterion(outputs, targets)

            loss = sum(losses[k] * criterion.weight_dict[k] for k in losses.keys() if k in criterion.weight_dict)

        # 反向传播
        scaler.scale(loss).backward()

        # 梯度裁剪
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)

        # 更新参数
        scaler.step(optimizer)
        scaler.update()

        # 记录损失
        total_loss += loss.item()
        for k, v in losses.items():
            if k not in losses_dict:
                losses_dict[k] = 0
            losses_dict[k] += v.item()

        # 更新进度条
        avg_loss = total_loss / (batch_idx + 1)
        progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})

    avg_loss = total_loss / len(dataloader)
    return avg_loss, {k: v / len(dataloader) for k, v in losses_dict.items()}


def evaluate(model, dataloader, criterion):
    """评估模型"""
    model.eval()
    total_loss = 0
    losses_dict = {}

    progress_bar = tqdm(dataloader, desc='[Eval]')

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(progress_bar):
            images = images.to(Config.DEVICE)
            targets = [{k: v.to(Config.DEVICE) for k, v in t.items()} for t in targets]

            outputs = model(images)
            losses = criterion(outputs, targets)

            loss = sum(losses[k] * criterion.weight_dict[k] for k in losses.keys() if k in criterion.weight_dict)

            total_loss += loss.item()
            for k, v in losses.items():
                if k not in losses_dict:
                    losses_dict[k] = 0
                losses_dict[k] += v.item()

            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})

    avg_loss = total_loss / len(dataloader)
    return avg_loss, {k: v / len(dataloader) for k, v in losses_dict.items()}


def print_training_info(train_loader, val_loader, model):
    """打印训练信息"""
    print("=" * 60)
    print("DETR 道路交通车辆检测模型训练")
    print("=" * 60)
    print(f"设备: {Config.DEVICE}")
    print(f"训练集大小: {len(train_loader.dataset)} 张图像")
    print(f"验证集大小: {len(val_loader.dataset)} 张图像")
    print(f"批次大小: {Config.BATCH_SIZE}")
    print(f"总参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("=" * 60)


def main():
    # 创建输出目录
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    # 获取数据加载器
    train_loader, val_loader = get_dataloaders()

    # 创建模型
    model = DETR().to(Config.DEVICE)

    # 打印训练信息
    print_training_info(train_loader, val_loader, model)

    # 创建损失函数
    criterion = build_criterion(Config.NUM_CLASSES).to(Config.DEVICE)

    # 创建优化器
    param_dicts = [
        {"params": [p for n, p in model.named_parameters()
                    if "backbone" not in n and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters()
                    if "backbone" in n and p.requires_grad],
         "lr": Config.LEARNING_RATE * 0.1},
    ]
    optimizer = optim.AdamW(param_dicts, lr=Config.LEARNING_RATE,
                            weight_decay=Config.WEIGHT_DECAY)

    # 创建学习率调度器
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)

    # 混合精度训练
    scaler = GradScaler()

    # 加载检查点
    start_epoch = 0
    best_loss = float('inf')
    best_model_path = Config.MODEL_SAVE_PATH.replace('.pth', '_best.pth')

    if os.path.exists(best_model_path):
        start_epoch, best_loss = load_checkpoint(
            model, optimizer, scheduler, best_model_path
        )

    # 训练循环
    for epoch in range(start_epoch, Config.NUM_EPOCHS):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{Config.NUM_EPOCHS}")
        print(f"{'=' * 60}")

        # 训练
        train_start_time = time.time()
        train_loss, train_losses = train_epoch(
            model, train_loader, criterion, optimizer, scaler, epoch + 1
        )
        train_time = time.time() - train_start_time

        # 评估
        val_start_time = time.time()
        val_loss, val_losses = evaluate(model, val_loader, criterion)
        val_time = time.time() - val_start_time

        # 更新学习率
        scheduler.step()

        # 打印详细损失信息
        print(f"\n训练统计:")
        print(f"  训练时间: {train_time:.1f}s, 验证时间: {val_time:.1f}s")
        print(f"  训练损失: {train_loss:.4f}")
        print(f"  验证损失: {val_loss:.4f}")

        for k, v in train_losses.items():
            print(f"  训练 {k}: {v:.4f}")

        for k, v in val_losses.items():
            print(f"  验证 {k}: {v:.4f}")

        # 保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss,
                Config.MODEL_SAVE_PATH, is_best=True
            )
            print(f"  ✓ 新的最佳模型! 验证损失: {val_loss:.4f}")
        else:
            print(f"  - 当前最佳验证损失: {best_loss:.4f}")

        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"{Config.OUTPUT_DIR}checkpoint_epoch_{epoch + 1}.pth"
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss,
                checkpoint_path, is_best=False
            )
            print(f"  ✓ 保存检查点到: {checkpoint_path}")

    # 保存最终模型
    save_checkpoint(
        model, optimizer, scheduler, Config.NUM_EPOCHS, val_loss,
        Config.MODEL_SAVE_PATH, is_best=False
    )

    print(f"\n{'=' * 60}")
    print("训练完成！")
    print(f"最佳验证损失: {best_loss:.4f}")
    print(f"模型保存到: {Config.OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()