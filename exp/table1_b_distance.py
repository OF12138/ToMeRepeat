import sys
import os
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import tome
from shared_eval import get_dataloader, validate, verify_dataset
from exp.mae_utils import load_mae_vit_large

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="/home/share/imagenet", help="Path to ImageNet dataset")
    args = parser.parse_args()

    # 1. 验证数据集
    verify_dataset(args.data_path)
    data_loader = get_dataloader(args.data_path, batch_size=32)
    
    results = {}
    for dist_func in ['eucl', 'cosine', 'dot', 'softmax']:
        print(f"\n--- Testing Distance Function: {dist_func} ---")
        
        # 2. 修改全局匹配计算距离算法
        ToMeConfig.distance_func = dist_func
        
        # 3. 创建测试模型
        model = timm.create_model("vit_large_patch16_224", pretrained=True)
        tome.patch.timm(model, prop_attn=False)
        model.r = 8
        
        acc, throughput = validate(model, data_loader)
        results[dist_func] = (acc, throughput)

    print("\n" + "="*50)
    print("Table 1(b): Distance Function Results")
    print("Function\tAcc (%)\tIm/s")
    for k, (acc, thro) in results.items():
        print(f"{k}\t{acc:.2f}\t{thro:.1f}")
    print("="*50)
