import argparse
import timm
import tome
from shared_eval import get_dataloader, validate, verify_dataset
from tome.config import ToMeConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="/home/share/imagenet", help="Path to ImageNet dataset")
    args = parser.parse_args()

    verify_dataset(args.data_path)
    data_loader = get_dataloader(args.data_path, batch_size=32)
    
    results = {}
    for style in ['sequential', 'alternating', 'random']:
        print(f"\n--- Testing Partition Style: {style} ---")
        
        ToMeConfig.partition_style = style
        
        model = timm.create_model("vit_large_patch16_224.mae", pretrained=True)
        tome.patch.timm(model, prop_attn=False)
        model.r = 8
        
        acc, throughput = validate(model, data_loader)
        results[style] = (acc, throughput)

    print("\n" + "="*50)
    print("Table 1(e): Partition Style Results")
    print("Function\tAcc (%)\tIm/s")
    for k, (acc, thro) in results.items():
        print(f"{k}\t{acc:.2f}\t{thro:.1f}")
    print("="*50)
