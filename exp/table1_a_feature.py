import sys
import os
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import tome
from shared_eval import get_dataloader, validate, verify_dataset
from mae_utils import load_mae_vit_large

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="/home/share/imagenet", help="Path to ImageNet dataset")
    args = parser.parse_args()

    # 1. 验证数据集
    verify_dataset(args.data_path)
    data_loader = get_dataloader(args.data_path, batch_size=256)
    
    results = {}
    for feature in ['Xpre', 'X', 'K', 'Q', 'V']:
        print(f"\n--- Testing Feature Choice: {feature} ---")
        
        # 2-3. 使用 MAE 加载器读取论文官方权重
        model = load_mae_vit_large()
        
        # 依据 MAE 默认无需 prop_attn
        tome.patch.timm(model, prop_attn=False)
        model.r = 8
        # 将配置参数直接穿透入底层的字典，拒绝 Python Import 开销
        model._tome_info["feature_choice"] = feature
        
        acc, throughput = validate(model, data_loader)
        results[feature] = (acc, throughput)
        
        del model
        torch.cuda.empty_cache()

    print("\n" + "="*50)
    print("Table 1(a): Feature Choice Results")
    print("Feature\tAcc (%)\tIm/s")
    for k, (acc, thro) in results.items():
        print(f"{k}\t{acc:.2f}\t{thro:.1f}")
    print("="*50)
