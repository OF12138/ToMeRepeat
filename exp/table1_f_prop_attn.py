import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import timm
import tome
from shared_eval import get_dataloader, validate, verify_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="/home/share/imagenet", help="Path to ImageNet dataset")
    args = parser.parse_args()

    verify_dataset(args.data_path)
    data_loader = get_dataloader(args.data_path, batch_size=32)
    
    results = {}
    print("\n--- Testing Proportional Attention ---")
    
    tests = [
        ('mae', 'vit_large_patch16_224', False),
        ('mae w/ prop', 'vit_large_patch16_224', True),
        ('augreg', 'vit_large_patch16_224', False),
        ('augreg w/ prop', 'vit_large_patch16_224', True),
    ]
    
    for name, model_name, prop_attn in tests:
        print(f"Running {name}...")
        model = timm.create_model(model_name, pretrained=True)
        tome.patch.timm(model, prop_attn=prop_attn)
        model.r = 8
        
        acc, throughput = validate(model, data_loader)
        results[name] = (acc, throughput)

    print("\n" + "="*50)
    print("Table 1(f): Proportional Attn Results")
    for name, _, _ in tests: print(f"{name}\t{results[name][0]:.2f}\t{results[name][1]:.1f}")
    print("="*50)
