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

    verify_dataset(args.data_path)
    data_loader = get_dataloader(args.data_path, batch_size=32)
    
    # === 表 1(c): 头聚会策略 ===
    print("\n" + "="*50)
    print("--- Testing Table 1(c) Head Aggregation ---")
    results_c = {}
    ToMeConfig.combine_method = 'weighted avg' # default
    for agg in ['concat', 'mean']:
        ToMeConfig.head_aggregation = agg
        
        model = timm.create_model("vit_large_patch16_224", pretrained=True)
        tome.patch.timm(model, prop_attn=False)
        model.r = 8
        
        acc, throughput = validate(model, data_loader)
        results_c[agg] = (acc, throughput)

    # === 表 1(d): 联合特征保留策略 ===
    print("\n" + "="*50)
    print("--- Testing Table 1(d) Combining Method ---")
    results_d = {}
    ToMeConfig.head_aggregation = 'mean' # reset to default
    for mode in ['keep one', 'max pool', 'avg pool', 'weighted avg']:
        if mode == 'weighted avg':
            results_d[mode] = results_c['mean']
            continue
            
        ToMeConfig.combine_method = mode
        
        model = timm.create_model("vit_large_patch16_224.mae", pretrained=True)
        tome.patch.timm(model, prop_attn=False)
        model.r = 8
        
        acc, throughput = validate(model, data_loader)
        results_d[mode] = (acc, throughput)

    print("\n" + "="*50)
    print("Table 1(c): Head Aggregation Results")
    for k, (acc, thro) in results_c.items(): print(f"{k}\t{acc:.2f}%\t{thro:.1f} im/s")
    
    print("\nTable 1(d): Combining Method Results")
    for k, (acc, thro) in results_d.items(): print(f"{k}\t{acc:.2f}%\t{thro:.1f} im/s")
    print("="*50)
