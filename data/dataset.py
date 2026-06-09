import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random


class FaceDataset(Dataset):
    """
    针对 CASIA 数据集优化的自定义数据集类
    数据集特点：500人，每人65张（5张原始+12种变体×5）
    优化策略：
    1. 训练集：原始图 + 85%变体（更多数据 = 更好泛化）
    2. 验证集：15%变体（模拟未见过的场景）
    3. 修复标签计算逻辑
    """

    def __init__(self, root_dir, is_train=True, train_ratio=0.85):
        self.root_dir = root_dir
        self.is_train = is_train
        self.image_paths = []
        self.labels = []
        self.class_names = sorted(os.listdir(root_dir))

        # 变体类型定义（根据实际文件名）
        variant_types = [
            '_block_eyes', '_block_mouth',
            '_blue', '_green', '_red', '_gray',
            '_brighter', '_darker',
            '_left_rotate45', '_right_rotate_45',
            '_blurred', '_vflip'
        ]

        # 构建索引映射 - 按人划分
        for label, class_name in enumerate(self.class_names):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                original_images = []
                variant_images = {t: [] for t in variant_types}

                # 分类所有图片
                for img_name in os.listdir(class_path):
                    if img_name.endswith(('.jpg', '.png')):
                        img_path = os.path.join(class_path, img_name)

                        # 判断是原始图还是变体
                        is_original = True
                        for variant in variant_types:
                            if variant in img_name:
                                variant_images[variant].append(img_path)
                                is_original = False
                                break
                        if is_original:
                            original_images.append(img_path)

                # 划分训练集和验证集
                random.seed(42 + label)
                n_orig_train = max(1, int(len(original_images) * train_ratio))
                orig_selected = random.sample(original_images, n_orig_train)
                train_images = list(orig_selected)
                val_images = [img for img in original_images if img not in orig_selected]

                for variant in variant_types:
                    random.seed(42 + label)
                    all_variants = variant_images[variant]
                    if len(all_variants) > 0:
                        n_train = max(1, int(len(all_variants) * train_ratio))
                        selected = random.sample(all_variants, n_train)
                        train_images.extend(selected)
                        val_images.extend([img for img in all_variants if img not in selected])

                # 确保每个类在验证集中至少有 1 张图
                if len(val_images) == 0 and len(train_images) > 1:
                    val_images.append(train_images.pop())

                if is_train:
                    self.image_paths.extend(train_images)
                    self.labels.extend([label] * len(train_images))
                else:
                    self.image_paths.extend(val_images)
                    self.labels.extend([label] * len(val_images))

        if is_train:
            self.transform = transforms.Compose([
                # 1. 尺度裁剪增强（保守：0.85-1.0，减少训练/验证分布差异）
                transforms.RandomResizedCrop(112, scale=(0.85, 1.0), ratio=(0.95, 1.05)),

                # 2. 水平翻转
                transforms.RandomHorizontalFlip(p=0.5),

                # 3. 旋转增强（±5°，数据集已含 45° 旋转变体）
                transforms.RandomRotation(degrees=5, expand=False, fill=0),

                # 4. 颜色增强（保守：数据集已含颜色变体）
                transforms.ColorJitter(
                    brightness=0.15, contrast=0.15,
                    saturation=0.15, hue=0.05
                ),

                # 6. 转张量
                transforms.ToTensor(),

                # 7. 标准化 (归一化到 [-1, 1])
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),

                # RandomErasing: value=0 对应归一化后的均值（灰色）
                transforms.RandomErasing(
                    p=0.15, scale=(0.02, 0.12),
                    ratio=(0.3, 3.3), value=0
                ),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(128),  # 先resize到稍大尺寸
                transforms.CenterCrop(112),  # 再中心裁剪，避免变形
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, label


def get_dataloader(root_dir, batch_size=96, num_workers=0):
    """
    创建训练集和验证集的 DataLoader
    """
    # Windows 系统建议使用单进程
    if os.name == 'nt' and num_workers > 0:
        print(f"⚠️  Windows 系统建议单进程，当前使用 {num_workers}")

    train_dataset = FaceDataset(root_dir, is_train=True)
    val_dataset = FaceDataset(root_dir, is_train=False)

    print(f"📊 训练集: {len(train_dataset)} 张 | 验证集: {len(val_dataset)} 张")

    # 使用 persistent_workers 提高性能
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              drop_last=True, prefetch_factor=4,
                              persistent_workers=(num_workers > 0))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            persistent_workers=(num_workers > 0))

    return train_loader, val_loader, len(train_dataset.class_names)