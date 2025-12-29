# -*- coding: utf-8 -*-
"""
自动生成所有模型的超参数搜索空间配置
"""
import os
import json

# 定义输出目录
OUTPUT_DIR = "PyPOTS_tuning_configs"

# 每个模型的超参数搜索空间
TUNING_SPACES = {
    'HELIX': {
        'ETT_h1': {
            "n_steps": {"_type": "choice", "_value": [48]},
            "n_features": {"_type": "choice", "_value": [7]},
            "epochs": {"_type": "choice", "_value": [1000]},
            "patience": {"_type": "choice", "_value": [10]},
            "pe_dim": {"_type": "choice", "_value": [12, 24, 48]},
            "feature_embed_dim": {"_type": "choice", "_value": [6, 12, 24]},
            "d_model": {"_type": "choice", "_value": [96, 128, 192, 256]},
            "n_heads": {"_type": "choice", "_value": [4, 6, 8]},
            "n_layers": {"_type": "choice", "_value": [1, 2, 3]},
            "dropout": {"_type": "choice", "_value": [0, 0.1, 0.2, 0.3]},
            "batch_size": {"_type": "choice", "_value": [8, 16, 32]},
            "lr": {"_type": "loguniform", "_value": [0.0001, 0.01]},
        },
        'PeMS': {
            "n_steps": {"_type": "choice", "_value": [24]},
            "n_features": {"_type": "choice", "_value": [862]},
            "epochs": {"_type": "choice", "_value": [1000]},
            "patience": {"_type": "choice", "_value": [10]},
            "pe_dim": {"_type": "choice", "_value": [6, 12, 24]},
            "feature_embed_dim": {"_type": "choice", "_value": [32, 64, 128]},
            "d_model": {"_type": "choice", "_value": [384, 512, 576, 768]},
            "n_heads": {"_type": "choice", "_value": [6, 8, 12]},
            "n_layers": {"_type": "choice", "_value": [1, 2, 3]},
            "dropout": {"_type": "choice", "_value": [0, 0.1, 0.2]},
            "batch_size": {"_type": "choice", "_value": [1, 2]},
            "lr": {"_type": "loguniform", "_value": [0.0001, 0.005]},
        }
    },
    
    'TEFN': {
        'ETT_h1': {
            "n_steps": {"_type": "choice", "_value": [48]},
            "n_features": {"_type": "choice", "_value": [7]},
            "epochs": {"_type": "choice", "_value": [1000]},
            "patience": {"_type": "choice", "_value": [10]},
            "n_fod": {"_type": "choice", "_value": [1, 2, 3]},
            "apply_nonstationary_norm": {"_type": "choice", "_value": [True, False]},
            "batch_size": {"_type": "choice", "_value": [8, 16, 32]},
            "lr": {"_type": "loguniform", "_value": [0.0001, 0.01]},
        },
        'PeMS': {
            "n_steps": {"_type": "choice", "_value": [24]},
            "n_features": {"_type": "choice", "_value": [862]},
            "epochs": {"_type": "choice", "_value": [1000]},
            "patience": {"_type": "choice", "_value": [10]},
            "n_fod": {"_type": "choice", "_value": [1, 2, 3]},
            "apply_nonstationary_norm": {"_type": "choice", "_value": [True, False]},
            "batch_size": {"_type": "choice", "_value": [1, 2]},
            "lr": {"_type": "loguniform", "_value": [0.0001, 0.005]},
        }
    },
    
    'TimeMixer': {
        'ETT_h1': {
            "n_steps": {"_type": "choice", "_value": [48]},
            "n_features": {"_type": "choice", "_value": [7]},
            "epochs": {"_type": "choice", "_value": [1000]},
            "patience": {"_type": "choice", "_value": [10]},
            "n_layers": {"_type": "choice", "_value": [1, 2, 3]},
            "d_model": {"_type": "choice", "_value": [32, 64, 128]},
            "d_ffn": {"_type": "choice", "_value": [64, 128, 256]},
            "top_k": {"_type": "choice", "_value": [3, 5, 7]},
            "dropout": {"_type": "choice", "_value": [0, 0.1, 0.2]},
            "channel_independence": {"_type": "choice", "_value": [True, False]},
            "decomp_method": {"_type": "choice", "_value": ["moving_avg"]},
            "moving_avg": {"_type": "choice", "_value": [5, 13, 25]},
            "downsampling_layers": {"_type": "choice", "_value": [1, 2]},
            "downsampling_window": {"_type": "choice", "_value": [2, 4]},
            "apply_nonstationary_norm": {"_type": "choice", "_value": [False]},
            "batch_size": {"_type": "choice", "_value": [8, 16, 32]},
            "lr": {"_type": "loguniform", "_value": [0.0001, 0.01]},
        },
        'PeMS': {
            "n_steps": {"_type": "choice", "_value": [24]},
            "n_features": {"_type": "choice", "_value": [862]},
            "epochs": {"_type": "choice", "_value": [1000]},
            "patience": {"_type": "choice", "_value": [10]},
            "n_layers": {"_type": "choice", "_value": [1, 2, 3]},
            "d_model": {"_type": "choice", "_value": [128, 256, 512]},
            "d_ffn": {"_type": "choice", "_value": [256, 512, 1024]},
            "top_k": {"_type": "choice", "_value": [3, 5, 7]},
            "dropout": {"_type": "choice", "_value": [0, 0.1, 0.2]},
            "channel_independence": {"_type": "choice", "_value": [True, False]},
            "decomp_method": {"_type": "choice", "_value": ["moving_avg"]},
            "moving_avg": {"_type": "choice", "_value": [5, 13, 25]},
            "downsampling_layers": {"_type": "choice", "_value": [1, 2]},
            "downsampling_window": {"_type": "choice", "_value": [2, 4]},
            "apply_nonstationary_norm": {"_type": "choice", "_value": [False]},
            "batch_size": {"_type": "choice", "_value": [1, 2]},
            "lr": {"_type": "loguniform", "_value": [0.0001, 0.005]},
        }
    },
    
    'ModernTCN': {
        'ETT_h1': {
            "n_steps": {"_type": "choice", "_value": [48]},
            "n_features": {"_type": "choice", "_value": [7]},
            "epochs": {"_type": "choice", "_value": [1000]},
            "patience": {"_type": "choice", "_value": [10]},
            "patch_size": {"_type": "choice", "_value": [4, 6, 8]},
            "patch_stride": {"_type": "choice", "_value": [4, 6, 8]},
            "downsampling_ratio": {"_type": "choice", "_value": [2, 4]},
            "ffn_ratio": {"_type": "choice", "_value": [2, 4]},
            # 必需参数 - 固定值
            "num_blocks": {"_type": "choice", "_value": [[1, 1]]},
            "large_size": {"_type": "choice", "_value": [[7, 7]]},
            "small_size": {"_type": "choice", "_value": [[3, 3]]},
            "dims": {"_type": "choice", "_value": [[32, 32], [64, 64], [32, 64]]},
            "small_kernel_merged": {"_type": "choice", "_value": [False]},
            "backbone_dropout": {"_type": "choice", "_value": [0, 0.1, 0.2]},
            "head_dropout": {"_type": "choice", "_value": [0, 0.1, 0.2]},
            "use_multi_scale": {"_type": "choice", "_value": [False]},
            "individual": {"_type": "choice", "_value": [False]},
            "apply_nonstationary_norm": {"_type": "choice", "_value": [False]},
            "batch_size": {"_type": "choice", "_value": [8, 16, 32]},
            "lr": {"_type": "loguniform", "_value": [0.0001, 0.01]},
        },
        'PeMS': {
            "n_steps": {"_type": "choice", "_value": [24]},
            "n_features": {"_type": "choice", "_value": [862]},
            "epochs": {"_type": "choice", "_value": [1000]},
            "patience": {"_type": "choice", "_value": [10]},
            "patch_size": {"_type": "choice", "_value": [4, 6, 8]},
            "patch_stride": {"_type": "choice", "_value": [4, 6, 8]},
            "downsampling_ratio": {"_type": "choice", "_value": [2, 4]},
            "ffn_ratio": {"_type": "choice", "_value": [2, 4]},
            # 必需参数 - 固定值
            "num_blocks": {"_type": "choice", "_value": [[1, 1]]},
            "large_size": {"_type": "choice", "_value": [[7, 7]]},
            "small_size": {"_type": "choice", "_value": [[3, 3]]},
            "dims": {"_type": "choice", "_value": [[64, 64], [128, 128], [64, 128]]},
            "small_kernel_merged": {"_type": "choice", "_value": [False]},
            "backbone_dropout": {"_type": "choice", "_value": [0, 0.1, 0.2]},
            "head_dropout": {"_type": "choice", "_value": [0, 0.1, 0.2]},
            "use_multi_scale": {"_type": "choice", "_value": [False]},
            "individual": {"_type": "choice", "_value": [False]},
            "apply_nonstationary_norm": {"_type": "choice", "_value": [False]},
            "batch_size": {"_type": "choice", "_value": [1, 2]},
            "lr": {"_type": "loguniform", "_value": [0.0001, 0.005]},
        }
    },
    
    'ImputeFormer': {
        'ETT_h1': {
            "n_steps": {"_type": "choice", "_value": [48]},
            "n_features": {"_type": "choice", "_value": [7]},
            "epochs": {"_type": "choice", "_value": [1000]},
            "patience": {"_type": "choice", "_value": [10]},
            "n_layers": {"_type": "choice", "_value": [1, 2, 3]},
            "d_input_embed": {"_type": "choice", "_value": [16, 32, 64]},
            "d_learnable_embed": {"_type": "choice", "_value": [16, 32, 64]},
            "d_proj": {"_type": "choice", "_value": [16, 32, 64]},
            "d_ffn": {"_type": "choice", "_value": [32, 64, 128]},
            "n_temporal_heads": {"_type": "choice", "_value": [2, 4, 8]},
            "dropout": {"_type": "choice", "_value": [0, 0.1, 0.2]},
            "batch_size": {"_type": "choice", "_value": [8, 16, 32]},
            "lr": {"_type": "loguniform", "_value": [0.0001, 0.01]},
        },
        'PeMS': {
            "n_steps": {"_type": "choice", "_value": [24]},
            "n_features": {"_type": "choice", "_value": [862]},
            "epochs": {"_type": "choice", "_value": [1000]},
            "patience": {"_type": "choice", "_value": [10]},
            "n_layers": {"_type": "choice", "_value": [1, 2, 3]},
            "d_input_embed": {"_type": "choice", "_value": [64, 128, 256]},
            "d_learnable_embed": {"_type": "choice", "_value": [64, 128, 256]},
            "d_proj": {"_type": "choice", "_value": [64, 128, 256]},
            "d_ffn": {"_type": "choice", "_value": [128, 256, 512]},
            "n_temporal_heads": {"_type": "choice", "_value": [4, 8, 16]},
            "dropout": {"_type": "choice", "_value": [0, 0.1, 0.2]},
            "batch_size": {"_type": "choice", "_value": [1, 2]},
            "lr": {"_type": "loguniform", "_value": [0.0001, 0.005]},
        }
    },
    
    'TOTEM': {
        'ETT_h1': {
            "n_steps": {"_type": "choice", "_value": [48]},
            "n_features": {"_type": "choice", "_value": [7]},
            "epochs": {"_type": "choice", "_value": [1000]},
            "patience": {"_type": "choice", "_value": [10]},
            "d_block_hidden": {"_type": "choice", "_value": [16, 32, 64]},
            "n_residual_layers": {"_type": "choice", "_value": [1, 2, 3]},
            "d_residual_hidden": {"_type": "choice", "_value": [8, 16, 32]},
            "d_embedding": {"_type": "choice", "_value": [16, 32, 64]},
            "n_embeddings": {"_type": "choice", "_value": [128, 256, 512]},
            "commitment_cost": {"_type": "choice", "_value": [0.1, 0.25, 0.5]},
            "compression_factor": {"_type": "choice", "_value": [2, 4, 8]},
            "batch_size": {"_type": "choice", "_value": [8, 16, 32]},
            "lr": {"_type": "loguniform", "_value": [0.0001, 0.01]},
        },
        'PeMS': {
            "n_steps": {"_type": "choice", "_value": [24]},
            "n_features": {"_type": "choice", "_value": [862]},
            "epochs": {"_type": "choice", "_value": [1000]},
            "patience": {"_type": "choice", "_value": [10]},
            "d_block_hidden": {"_type": "choice", "_value": [64, 128, 256]},
            "n_residual_layers": {"_type": "choice", "_value": [1, 2, 3]},
            "d_residual_hidden": {"_type": "choice", "_value": [32, 64, 128]},
            "d_embedding": {"_type": "choice", "_value": [64, 128, 256]},
            "n_embeddings": {"_type": "choice", "_value": [256, 512, 1024]},
            "commitment_cost": {"_type": "choice", "_value": [0.1, 0.25, 0.5]},
            "compression_factor": {"_type": "choice", "_value": [2, 4, 8]},
            "batch_size": {"_type": "choice", "_value": [1, 2]},
            "lr": {"_type": "loguniform", "_value": [0.0001, 0.005]},
        }
    },
    
    'TimeMixerPP': {
        'ETT_h1': {
            "n_steps": {"_type": "choice", "_value": [48]},
            "n_features": {"_type": "choice", "_value": [7]},
            "epochs": {"_type": "choice", "_value": [1000]},
            "patience": {"_type": "choice", "_value": [10]},
            "n_layers": {"_type": "choice", "_value": [1, 2, 3]},
            "d_model": {"_type": "choice", "_value": [32, 64, 128]},
            "d_ffn": {"_type": "choice", "_value": [64, 128, 256]},
            "top_k": {"_type": "choice", "_value": [3, 5, 7]},
            "n_heads": {"_type": "choice", "_value": [2, 4, 8]},
            "n_kernels": {"_type": "choice", "_value": [4, 6, 8]},
            "dropout": {"_type": "choice", "_value": [0, 0.1, 0.2]},
            "channel_mixing": {"_type": "choice", "_value": [True, False]},
            "channel_independence": {"_type": "choice", "_value": [True, False]},
            "downsampling_layers": {"_type": "choice", "_value": [1, 2]},
            "downsampling_window": {"_type": "choice", "_value": [2, 4]},
            "apply_nonstationary_norm": {"_type": "choice", "_value": [False]},
            "batch_size": {"_type": "choice", "_value": [8, 16, 32]},
            "lr": {"_type": "loguniform", "_value": [0.0001, 0.01]},
        }
    },
    
    'TimeLLM': {
        'ETT_h1': {
            "n_steps": {"_type": "choice", "_value": [48]},
            "n_features": {"_type": "choice", "_value": [7]},
            "epochs": {"_type": "choice", "_value": [1000]},
            "patience": {"_type": "choice", "_value": [10]},
            # 必需参数 - 固定值
            "llm_model_type": {"_type": "choice", "_value": ["BERT"]},
            "d_llm": {"_type": "choice", "_value": [768]},
            "domain_prompt_content": {"_type": "choice", "_value": ["Electricity transformer temperature time series data"]},
            # 可调参数
            "n_layers": {"_type": "choice", "_value": [1, 2, 3]},
            "patch_size": {"_type": "choice", "_value": [8, 12, 16]},
            "patch_stride": {"_type": "choice", "_value": [8, 12, 16]},
            "d_model": {"_type": "choice", "_value": [16, 32, 64]},
            "d_ffn": {"_type": "choice", "_value": [32, 64, 128]},
            "n_heads": {"_type": "choice", "_value": [2, 4, 8]},
            "dropout": {"_type": "choice", "_value": [0, 0.1, 0.2]},
            "batch_size": {"_type": "choice", "_value": [8, 16, 32]},
            "lr": {"_type": "loguniform", "_value": [0.00005, 0.001]},
        }
    },
    
    'MOMENT': {
        'ETT_h1': {
            "n_steps": {"_type": "choice", "_value": [48]},
            "n_features": {"_type": "choice", "_value": [7]},
            "epochs": {"_type": "choice", "_value": [1000]},
            "patience": {"_type": "choice", "_value": [10]},
            "patch_size": {"_type": "choice", "_value": [8, 12, 16]},
            "patch_stride": {"_type": "choice", "_value": [8, 12, 16]},
            # 必需参数 - 固定值
            "transformer_backbone": {"_type": "choice", "_value": ["t5-base"]},
            "transformer_type": {"_type": "choice", "_value": ["encoder_only"]},
            "d_model": {"_type": "choice", "_value": [768]},
            "revin_affine": {"_type": "choice", "_value": [True]},
            "add_positional_embedding": {"_type": "choice", "_value": [True]},
            "value_embedding_bias": {"_type": "choice", "_value": [True]},
            "orth_gain": {"_type": "choice", "_value": [1.41]},
            # 可调参数
            "n_layers": {"_type": "choice", "_value": [2, 4, 6]},
            "d_ffn": {"_type": "choice", "_value": [1024, 2048, 4096]},
            "dropout": {"_type": "choice", "_value": [0, 0.1, 0.2]},
            "head_dropout": {"_type": "choice", "_value": [0, 0.1, 0.2]},
            "finetuning_mode": {"_type": "choice", "_value": ["linear-probing", "end-to-end"]},
            "mask_ratio": {"_type": "choice", "_value": [0.1, 0.3, 0.5]},
            "batch_size": {"_type": "choice", "_value": [8, 16, 32]},
            "lr": {"_type": "loguniform", "_value": [0.00005, 0.001]},
        }
    },
}


def generate_all_configs():
    """生成所有配置文件"""
    
    for model_name, datasets in TUNING_SPACES.items():
        model_dir = os.path.join(OUTPUT_DIR, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        for dataset_name, tuning_space in datasets.items():
            filename = f"{model_name}_{dataset_name}_tuning_space.json"
            filepath = os.path.join(model_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(tuning_space, f, indent=2)
            
            print(f"✓ Generated: {filepath}")
    
    print(f"\n总共生成了配置文件")


if __name__ == "__main__":
    generate_all_configs()