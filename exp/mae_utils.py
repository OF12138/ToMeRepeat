import os
import torch
import torch.nn as nn
import timm

def patch_global_pool(model):
    """
    修改 timm 原生的类 token 分类方式，改为 MAE 使用的全局平均池化 (GAP) 分类方式。
    """
    # 如果模型没有 fc_norm（MAE引入的），我们给它临时加一个
    if not hasattr(model, 'fc_norm'):
        model.fc_norm = nn.LayerNorm(model.embed_dim, eps=1e-6)
        
    old_forward_features = model.forward_features
    
    def global_pool_forward_features(x):
        x = model.patch_embed(x)
        
        T = x.shape[1]  # 记录原始 patch 数量（196），用于后续比例池化
        
        cls_token = model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = model.pos_drop(x + model.pos_embed)
        x = model.blocks(x)
        
        # MAE 的全局平均池化（GAP），必须按 token size 加权！
        # 当 ToMe 合并 token 后，某些 token 代表了多个原始 patch，
        # 必须按 size 加权求和再除以总 patch 数 T，才能得到正确的全局均值
        if hasattr(model, '_tome_info') and model._tome_info.get("size") is not None:
            x = (x * model._tome_info["size"])[:, 1:, :].sum(dim=1) / T
        else:
            x = x[:, 1:, :].mean(dim=1)  # 无 ToMe 时直接取均值
        
        x = model.fc_norm(x)
        return x
        
    # 执行动态绑定替换
    model.forward_features = global_pool_forward_features

def load_mae_vit_large(weight_path="model/mae_finetuned_vit_large.pth", num_classes=1000):
    """
    利用预先存放在 Model/ 里的 MAE Finetuned 权重生成完美对齐的 vit_large_patch16_224
    """
    print(f"[MAE Loader] Instantiating vit_large_patch16_224...")
    # 注意，由于我们自己供血，pretrained=False
    model = timm.create_model("vit_large_patch16_224", pretrained=False, num_classes=num_classes)
    
    # 注入 Global Pool 能力
    patch_global_pool(model)
    
    weight_path = os.path.abspath(weight_path)
    if os.path.exists(weight_path):
        print(f"[MAE Loader] Found weights at {weight_path}, loading into model...")
        checkpoint = torch.load(weight_path, map_location='cpu')
        
        # 兼容一下各种可能的外壳
        state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
        
        new_state_dict = {}
        for k, v in state_dict.items():
            # 有时被打包在 model. 前缀下
            if k.startswith('model.'):
                k = k[6:]
            new_state_dict[k] = v
            
        # load strict=False 是为了避开原本模型里某些不用参与池化的原版 norm 和 cls
        msg = model.load_state_dict(new_state_dict, strict=False)
        print(f"[MAE Loader] Load Result: Missing keys: {len(msg.missing_keys)}, Unexpected keys: {len(msg.unexpected_keys)}")
    else:
        print(f"[MAE Loader] WARNING: {weight_path} Not Found! Accuracy will be ~0.1%!")
        
    return model
