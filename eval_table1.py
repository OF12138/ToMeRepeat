import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import tome

def validate(model, data_loader, device="cuda"):
    model.eval()
    model.to(device)
    
    correct = 0
    total = 0
    
    # 预热 GPU
    print("Pre-warming GPU...")
    dummy_input = torch.randn(16, 3, 224, 224).to(device)
    with torch.no_grad():
        for _ in range(10):
            model(dummy_input)
            
    torch.cuda.synchronize()
    start_time = time.time()
    
    # 正式推理
    print("Evaluating over validation set...")
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if total % 1000 == 0:
                print(f"Processed {total} images... Current Acc: {100 * correct / total:.2f}%")
                
    torch.cuda.synchronize()
    end_time = time.time()
    
    acc = 100 * correct / total
    throughput = total / (end_time - start_time)
    
    return acc, throughput

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Reproduce Table 1 of ToMe")
    parser.add_argument("--data-path", type=str, required=True, help="Path to ImageNet validation set directory")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--r", type=int, default=8, help="Number of tokens to reduce per layer (r=8 in Table 1)")
    parser.add_argument("--disable-tome", action='store_true', help="Run baseline without ToMe")
    args = parser.parse_args()

    # 1. 准备 ImageNet 的标准化 transform (参照 MAE 和 timm 的标准配置)
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=3), # bicubic
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # 2. 加载数据集
    val_dataset = datasets.ImageFolder(os.path.join(args.data_path, 'val'), transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # 3. 加载预训练的 MAE ViT-Large 模型
    # 注意：表 1 用的是 "ViT-L/16 from MAE"，timm 中的名称为 `vit_large_patch16_224.mae`
    print("Loading ViT-L/16 (MAE) model...")
    model = timm.create_model("vit_large_patch16_224.mae", pretrained=True)
    
    if not args.disable_tome:
        print(f"Applying ToMe patch with r={args.r}...")
        # 表 1: MAE 模型在评估时，论文中说明要关闭 proportional attention 设置 (prop_attn=False)
        tome.patch.timm(model, prop_attn=False)
        model.r = args.r
    else:
        print("Running WITHOUT ToMe...")
        
    print(f"Total batches to process: {len(val_loader)}")
    acc, throughput = validate(model, val_loader)
    
    print("-" * 50)
    print(f"Results:")
    print(f"Top-1 Accuracy: {acc:.2f}%")
    print(f"Throughput:     {throughput:.2f} im/s")
    print("-" * 50)
