import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm

def verify_dataset(path):
    """
    数据集校验函数
    作用：判断指定路径是否存在，确保 ImageNet 的结构满足 train/ 和 val/ 格式，
    并且确保它包含类别文件夹（正常 ImageNet-1k 应该有 1000 个文件夹）。
    """
    if not os.path.exists(path):
        # 抛出异常：找不到指定的总体路径
        raise FileNotFoundError(f"[Error] Dataset path {path} does not exist!")
        
    val_path = os.path.join(path, "val")
    if not os.path.exists(val_path):
        # 抛出异常：如果找不到验证集文件夹，说明数据集可能解压错位
        raise FileNotFoundError(f"[Error] Validation directory not found at {val_path}. Make sure it has train/ and val/ folders.")
    
    # 统计验证集目录下有多少个文件夹（即多少个类别）
    classes = [d for d in os.listdir(val_path) if os.path.isdir(os.path.join(val_path, d))]
    
    if len(classes) == 0:
         # 抛出异常：验证集文件夹为空，无法做精度测试
         raise FileNotFoundError(f"[Error] Validation directory {val_path} is empty or has no subdirectories.")
         
    if len(classes) < 100:
        # 如果类别少于 100，给出警告 (ImageNet-1k 有 1000 个类)
        print(f"[Warning] Found only {len(classes)} classes. ImageNet-1k should have 1000.")
    
    # 全部通过后输出成功日志
    print(f"[Dataset OK] Verified validation directory at {val_path} containing {len(classes)} classes.")

def get_dataloader(data_path, batch_size=256):
    """
    配置数据加载器 (DataLoader) 函数
    作用：定义模型的图像预处理流（裁剪、缩放、标准化），并将文件夹封装为 DataLoader，以支持快速批量提取。
    """
    # 标准的 ImageNet 推理预处理流程
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=3), # 将图像短边缩放至 256 (BICUBIC 插值)
        transforms.CenterCrop(224),              # 从中心裁剪出 224x224 分辨率
        transforms.ToTensor(),                   # 转换为 PyTorch 张量并且归一化到 [0, 1] 范围
        # 使用 ImageNet 的均值 (mean) 和标准差 (std) 进行通道特征的分布标准化
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # 实例化数据集对象
    val_dataset = datasets.ImageFolder(os.path.join(data_path, 'val'), transform=transform)
    # 创建 DataLoader：由于 A40 算力极强，将进图子进程飙升至 8 (num_workers=8) 喂饱显卡
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    return val_loader

def validate(model, data_loader, device="cuda", print_freq=50):
    """
    验证函数 (核心跑分逻辑)
    作用：对准备好的 DataLoader 和 Model 执行批量推理。
    它会统计分类准确率 (Accuracy)，并精准计算模型吞吐量 (Throughput，即 Images/s)
    """
    # 切换至推理模式，关闭 Dropout 或 BatchNorm 之类的随机截断
    model.eval()
    model.to(device)
    
    correct = 0 # 记录识别正确的总图像数量
    total = 0   # 记录跑过的总图像数量
    
    # 【GPU 预热机制】：如果不提前让 GPU 跑几次空数据，CUDA 会把模型初始化的耗时也算进去，导致速度测算不准确
    print("Pre-warming GPU...")
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    with torch.no_grad():
        for _ in range(3): 
            model(dummy_input) # 拉起几次空跑激活 GPU 的最高频率
            
    # 同步等待 GPU 所有流运行结束，确立正式开始时间点
    torch.cuda.synchronize()
    start_time = time.time()
    
    print("Evaluating...")
    # torch.no_grad(): 关闭自动求导反向传播缓存，显著节省显存并加快速度
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            # 将图与标签推入显卡显存
            images = images.to(device)
            labels = labels.to(device)
            
            # 模型前向传播，因为植入了 ToMe 的机制，其内部 Token 量会因为不断 merge 而自动消减
            outputs = model(images)
            
            # 寻找最高激活概率的那个神经元序号，作为预测结果 (predicted)
            _, predicted = torch.max(outputs, 1)
            
            # 统计数量
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 按预先配置的间隔刷新日志，打印当前进度与平均准确率
            if i % print_freq == 0 and i > 0:
                print(f"Processed {total} images... Current Acc: {100 * correct / total:.2f}%")
                
    # 必须在这个指令上同步并阻塞等待 GPU 的所有推理任务完毕，才能精准掐表结速
    torch.cuda.synchronize()
    end_time = time.time()
    
    # 计算评估结果：正确率 (Acc%) 和 每秒处理图像帧数 (Im/s)
    acc = 100 * correct / total
    throughput = total / (end_time - start_time)
    
    return acc, throughput
