import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm

def verify_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[Error] Dataset path {path} does not exist!")
    val_path = os.path.join(path, "val")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"[Error] Validation directory not found at {val_path}. Make sure it has train/ and val/ folders.")
    
    classes = [d for d in os.listdir(val_path) if os.path.isdir(os.path.join(val_path, d))]
    
    if len(classes) == 0:
         raise FileNotFoundError(f"[Error] Validation directory {val_path} is empty or has no subdirectories.")
         
    if len(classes) < 100:
        print(f"[Warning] Found only {len(classes)} classes. ImageNet-1k should have 1000.")
    print(f"[Dataset OK] Verified validation directory at {val_path} containing {len(classes)} classes.")

def get_dataloader(data_path, batch_size=64):
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_dataset = datasets.ImageFolder(os.path.join(data_path, 'val'), transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return val_loader

def validate(model, data_loader, device="cuda", print_freq=50):
    model.eval()
    model.to(device)
    
    correct = 0
    total = 0
    
    print("Pre-warming GPU...")
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    with torch.no_grad():
        for _ in range(3): 
            model(dummy_input)
            
    torch.cuda.synchronize()
    start_time = time.time()
    
    print("Evaluating...")
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % print_freq == 0 and i > 0:
                print(f"Processed {total} images... Current Acc: {100 * correct / total:.2f}%")
                
    torch.cuda.synchronize()
    end_time = time.time()
    
    acc = 100 * correct / total
    throughput = total / (end_time - start_time)
    
    return acc, throughput
