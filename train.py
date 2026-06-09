import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os
import glob
import math
from ArcFace.model import FaceEncoder, ArcMarginProduct
from data.dataset import get_dataloader


def validate(model, val_loader, device, margin_loss, criterion=None):
    """
    验证函数：使用 ArcFace margin_loss 权重进行分类，同时计算损失
    返回 (准确率%, 平均损失) 以便早停使用
    """
    model.eval()
    margin_loss.eval()

    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            embeddings = model(images)

            # 分类准确率
            cosine = F.linear(F.normalize(embeddings), F.normalize(margin_loss.weight))
            _, predicted = cosine.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 损失（仅在提供了 criterion 时计算）
            if criterion is not None:
                outputs = margin_loss(embeddings, labels)
                total_loss += criterion(outputs, labels).item()

    acc = (100. * correct / total) if total > 0 else 0.0
    avg_loss = total_loss / len(val_loader) if criterion is not None and len(val_loader) > 0 else 0.0
    return acc, avg_loss


def train_arcface_model(data_root='D:/face_dataset/Cleaned_CASIA_FaceV5',
                        epochs=150,
                        batch_size=128,
                        resume_from=None,
                        val_interval=4):
    """
    ArcFace 模型训练函数 — 使用 Warmup + StepLR + 混合精度 + 早停
    """
    # ========== 1. 设备配置 ==========
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[设备] {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"[显卡] {gpu_name} | 显存: {gpu_mem:.2f} GB")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # ========== 2. 数据加载 ==========
    num_workers = 4
    train_loader, val_loader, num_classes = get_dataloader(
        data_root, batch_size=batch_size, num_workers=num_workers
    )
    print(f"[数据] 类别数: {num_classes} | 进程数: {num_workers} | 批次大小: {batch_size}")

    # ========== 3. 模型初始化 ==========
    model = FaceEncoder(embedding_dim=512, use_se=True, use_sa=True).to(device)
    margin_loss = ArcMarginProduct(in_features=512, out_features=num_classes, s=30.0, m=0.50).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[模型] 总参数: {total_params/1e6:.2f}M | 可训练: {trainable_params/1e6:.2f}M")

    # ========== 4. 优化器 ==========
    base_lr = 0.01  # SGD 标准学习率
    optimizer = optim.SGD(
        [{'params': model.parameters()}, {'params': margin_loss.parameters()}],
        lr=base_lr, momentum=0.9, weight_decay=5e-4
    )

    # StepLR: 每 40 轮学习率衰减为原来的 0.1
    warmup_epochs = 5
    base_scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=40, gamma=0.1
    )

    # ========== 4. Warmup + StepLR 学习率调度 ==========
    class WarmupScheduler:
        def __init__(self, optimizer, base_scheduler, warmup_epochs, base_lr, start_epoch=0):
            self.optimizer = optimizer
            self.base_scheduler = base_scheduler
            self.warmup_epochs = warmup_epochs
            self.base_lr = base_lr
            self.current_epoch = start_epoch

            if self.current_epoch < self.warmup_epochs:
                lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
                for pg in self.optimizer.param_groups:
                    pg['lr'] = lr

        def step(self):
            self.current_epoch += 1
            if self.current_epoch < self.warmup_epochs:
                lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
                for pg in self.optimizer.param_groups:
                    pg['lr'] = lr
            else:
                self.base_scheduler.step()

    # ========== 5. 混合精度 ==========
    from torch.amp import GradScaler
    scaler = GradScaler() if device.type == 'cuda' else None
    autocast_enabled = device.type == 'cuda'

    criterion = nn.CrossEntropyLoss()

    # ========== 6. 断点续训 ==========
    start_epoch = 0
    best_val_loss = float('inf')
    best_acc = 0.0
    patience = 40
    no_improve_count = 0
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    if resume_from is None:
        ckpts = glob.glob(os.path.join(checkpoint_dir, 'latest_checkpoint.pth'))
        if ckpts:
            resume_from = ckpts[0]

    if resume_from and os.path.exists(resume_from):
        try:
            ckpt = torch.load(resume_from, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            margin_loss.load_state_dict(ckpt['margin_loss_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_acc = ckpt.get('best_acc', 0.0)
            best_val_loss = ckpt.get('best_val_loss', float('inf'))
            no_improve_count = ckpt.get('no_improve_count', 0)
            print(f"[恢复] 从第 {start_epoch} 轮继续 | 最佳准确率: {best_acc:.2f}% | 最佳损失: {best_val_loss:.4f}")
        except Exception as e:
            print(f"[警告] 检查点加载失败: {e}，将从头开始训练")

    scheduler = WarmupScheduler(optimizer, base_scheduler, warmup_epochs, base_lr, start_epoch=start_epoch)

    # ========== 7. 训练循环 ==========
    for epoch in range(start_epoch, epochs):
        scheduler.step()
        model.train()
        margin_loss.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f'第 {epoch+1}/{epochs} 轮', dynamic_ncols=True)
        for images, labels in loop:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                embeddings = model(images)
                outputs = margin_loss(embeddings, labels)
                loss = criterion(outputs, labels)

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
                torch.nn.utils.clip_grad_norm_(margin_loss.parameters(), max_norm=3.0)
                grad_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.norm().item() ** 2
                grad_norm = grad_norm ** 0.5
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
                torch.nn.utils.clip_grad_norm_(margin_loss.parameters(), max_norm=3.0)
                grad_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.norm().item() ** 2
                grad_norm = grad_norm ** 0.5
                optimizer.step()

            running_loss += loss.item()
            total += labels.size(0)

            # 训练准确率（只用于显示，不用于早停）
            with torch.no_grad():
                cos = F.linear(F.normalize(embeddings.detach()), F.normalize(margin_loss.weight))
                _, pred = cos.max(1)
                correct += pred.eq(labels).sum().item()

            current_acc = 100. * correct / total if total > 0 else 0.0
            loop.set_postfix(损失=f'{loss.item():.4f}', 准确率=f'{current_acc:.1f}%', 梯度范数=f'{grad_norm:.1f}')

        # 轮次统计
        train_acc = 100. * correct / total if total > 0 else 0.0
        avg_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        current_lr = optimizer.param_groups[0]['lr']

        # ========== 8. 验证 ==========
        val_acc = None
        val_loss = None
        if (epoch + 1) % val_interval == 0 or epoch == epochs - 1:
            val_acc, val_loss = validate(model, val_loader, device, margin_loss, criterion)
            model.train()
            margin_loss.train()

            gap = train_acc - val_acc
            status = '✅' if gap < 10 else ('⚠️' if gap < 20 else '🔴')
            print(f"[第 {epoch+1:3d} 轮] 训练: {train_acc:.2f}% | 验证: {val_acc:.2f}% | "
                  f"差距: {gap:.1f}% {status} | 损失: {avg_loss:.4f}/{val_loss:.4f} | 学习率: {current_lr:.2e}")

            # 早停：以验证损失为主指标
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                best_acc = val_acc
                no_improve_count = 0
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_face_model.pth'))
                print(f"  新最佳模型！验证损失: {val_loss:.4f} | 准确率: {val_acc:.2f}%")
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    print(f"[早停] 连续 {patience} 轮验证损失未改善，停止训练")
                    break
        else:
            print(f"[第 {epoch+1:3d} 轮] 训练: {train_acc:.2f}% | 损失: {avg_loss:.4f} | "
                  f"学习率: {current_lr:.2e} (跳过验证)")

        # ========== 9. 保存检查点 ==========
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'margin_loss_state_dict': margin_loss.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
            'best_val_loss': best_val_loss,
            'no_improve_count': no_improve_count,
        }, os.path.join(checkpoint_dir, 'latest_checkpoint.pth'))

    print(f"\n[完成] 最佳验证准确率: {best_acc:.2f}% | 最佳验证损失: {best_val_loss:.4f}")
    print(f"[模型] checkpoints/best_face_model.pth")


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

    try:
        train_arcface_model()
    except KeyboardInterrupt:
        print("\n[中断] 训练被用户终止")
    except Exception as e:
        print(f"\n[错误] {e}")
        import traceback
        traceback.print_exc()