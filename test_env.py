import torch
import timm
# import sys
# sys.path.append(r"E:\Code\Generation\ToME")
import tome
import time

def benchmark(model, device="cuda", img_size=224, batch_size=64, runs=50):
    model = model.eval().to(device)
    x = torch.rand(batch_size, 3, img_size, img_size, device=device)
    
    # 预热 GPU
    with torch.no_grad():
        for _ in range(10): 
            model(x)
            
    torch.cuda.synchronize()
    start_time = time.time()
    
    # 正式测试
    with torch.no_grad():
        for _ in range(runs):
            model(x)
            
    torch.cuda.synchronize()
    end_time = time.time()
    
    # 返回每秒处理多少张图片
    return (batch_size * runs) / (end_time - start_time)

if __name__ == "__main__":
    # 1. 加载原生 ViT-Base 模型
    print("Loading original ViT-Base...")
    model_name = "vit_base_patch16_224"
    model = timm.create_model(model_name, pretrained=False)
    
    # 测试原模型吞吐量
    throughput_orig = benchmark(model)
    print(f"Original Throughput: {throughput_orig:.2f} im/s")
    
    # 2. 注入 ToMe 的黑科技！只需要下面这两句话
    print("\nPatching model with ToMe...")
    tome.patch.timm(model)
    model.r = 16 # 每层移除(融合) 16 个 token (这个数字可以自行调整进行精度换速度的评估)
    
    # 测试打完补丁的模型吞吐量
    throughput_tome = benchmark(model)
    print(f"ToMe Throughput: {throughput_tome:.2f} im/s")
    print(f"Speedup: {throughput_tome / throughput_orig:.2f}x")
