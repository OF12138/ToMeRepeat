import os
import torch
from torchvision import datasets, transforms

# 数据集路径（直接使用共享目录，无需复制）
DATA_ROOT = "/home/share/imagenet"

# 标准 ImageNet 预处理（与 ToMe 论文一致）
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载数据集
train_dataset = datasets.ImageFolder(os.path.join(DATA_ROOT, "train"), transform=train_transform)
val_dataset = datasets.ImageFolder(os.path.join(DATA_ROOT, "val"), transform=val_transform)

# 创建 DataLoader（注意：由于是共享数据集，建议 num_workers 不要设置太大，避免影响他人）
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
)

# 测试数据加载
print(f"训练集大小: {len(train_dataset)}")
print(f"验证集大小: {len(val_dataset)}")

# 读取一个 batch 测试
images, labels = next(iter(val_loader))
print(f"Batch 形状: {images.shape}, 标签形状: {labels.shape}")