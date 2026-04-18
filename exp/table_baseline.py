import sys
import os

# 将根目录添加到 sys.path 以便能够导入 tome 和 exp
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import tome
from shared_eval import verify_dataset, get_dataloader, validate
from mae_utils import load_mae_vit_large

def main():
    data_path = "/home/share/imagenet"
    
    print("Verifying dataset...")
    verify_dataset(data_path)
    
    print("Loading data...")
    val_loader = get_dataloader(data_path, batch_size=256)
    
    results = []

    # ==========================================
    # 实验 1: 原始模型 (不使用 ToMe)
    # ==========================================
    print("\n--- Testing Baseline (No ToMe) ---")
    model_baseline = load_mae_vit_large()
    # 确保不开 ToMe
    
    acc_base, ims_base = validate(model_baseline, val_loader)
    results.append(("Baseline (No ToMe)", acc_base, ims_base))
    
    # 释放显存
    del model_baseline
    torch.cuda.empty_cache()

    # ==========================================
    # 实验 2: 使用 ToMe 模型 (r=8)
    # ==========================================
    print("\n--- Testing ToMe (r=8, prop_attn=False) ---")
    model_tome = load_mae_vit_large()
    
    # 注入 ToMe, 依据论文对于 Off-the-shelf MAE 模型，需要关闭 prop_attn 才能发挥威力
    tome.patch.timm(model_tome, prop_attn=False)
    model_tome.r = 8
    
    acc_tome, ims_tome = validate(model_tome, val_loader)
    results.append(("ToMe (r=8)", acc_tome, ims_tome))

    # ==========================================
    # 输出结果表格
    # ==========================================
    print("\n==================================================")
    print("Baseline vs ToMe Comparison")
    print(f"{'Method':<25} {'Acc (%)':<10} {'Im/s':<10}")
    for method, acc, ims in results:
        print(f"{method:<25} {acc:<10.2f} {ims:<10.1f}")
    print("==================================================\n")

if __name__ == '__main__':
    main()
