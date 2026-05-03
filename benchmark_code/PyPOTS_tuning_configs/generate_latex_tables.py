# -*- coding: utf-8 -*-
"""
生成超参数搜索空间的LaTeX表格
目标：ICML Best Paper级别的细致程度
"""
import os
import json

# 输出目录
OUTPUT_DIR = "/home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/latex_tables"

# 数据集顺序
DATASET_ORDER = ['ETT_h1', 'BeijingAir', 'ItalyAir', 'PeMS', 'PhysioNet2012']
DATASET_DISPLAY = {
    'ETT_h1': 'ETT-h1',
    'BeijingAir': 'BeijingAir', 
    'ItalyAir': 'ItalyAir',
    'PeMS': 'PeMS',
    'PhysioNet2012': 'PhysioNet2012'
}

# 参数分组定义
DATASET_PROPERTIES = ['n_steps', 'n_features']
TRAINING_CONFIG = ['epochs', 'patience', 'batch_size', 'lr']
LOSS_WEIGHTS = ['ORT_weight', 'MIT_weight']
# 其余参数自动归入 Model Architecture

# 模型顺序（HELIX变体将在脚注说明）
MODEL_ORDER = [
    'HELIX',
    'TEFN', 
    'TimeMixer',
    'ModernTCN', 
    'ImputeFormer',
    'TOTEM',
    'TimeMixerPP',
    'TimeLLM',
    'MOMENT'
]

# HELIX变体列表（共享相同搜索空间）
HELIX_VARIANTS = ['HELIX_NoSinusoidalPE', 'HELIX_NoFeatureEmbed', 'HELIX_NoHybrid', 'HELIX_NoFusion']

# 从generate_tuning_configs.py复制的完整配置
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
        },
        'BeijingAir': {
            "n_steps": {"_type": "choice", "_value": [24]},
            "n_features": {"_type": "choice", "_value": [132]},
            "epochs": {"_type": "choice", "_value": [1000]},
            "patience": {"_type": "choice", "_value": [10]},
            "pe_dim": {"_type": "choice", "_value": [6, 12, 24]},
            "feature_embed_dim": {"_type": "choice", "_value": [16, 24, 32]},
            "d_model": {"_type": "choice", "_value": [192, 256, 384]},
            "n_heads": {"_type": "choice", "_value": [4, 8, 12]},
            "n_layers": {"_type": "choice", "_value": [1, 2, 3]},
            "dropout": {"_type": "choice", "_value": [0, 0.1, 0.2]},
            "ORT_weight": {"_type": "choice", "_value": [1.0]},
            "MIT_weight": {"_type": "choice", "_value": [1.0]},
            "batch_size": {"_type": "choice", "_value": [4, 8, 16]},
            "lr": {"_type": "loguniform", "_value": [0.0001, 0.01]},
        },
        'PhysioNet2012': {
            "n_steps": {"_type": "choice", "_value": [48]},
            "n_features": {"_type": "choice", "_value": [35]},
            "epochs": {"_type": "choice", "_value": [1000]},
            "patience": {"_type": "choice", "_value": [10]},
            "pe_dim": {"_type": "choice", "_value": [8, 12, 16]},
            "feature_embed_dim": {"_type": "choice", "_value": [8, 10, 16]},
            "d_model": {"_type": "choice", "_value": [64, 96, 128]},
            "n_heads": {"_type": "choice", "_value": [2, 4, 8]},
            "n_layers": {"_type": "choice", "_value": [1, 2, 3]},
            "dropout": {"_type": "choice", "_value": [0, 0.1, 0.2]},
            "ORT_weight": {"_type": "choice", "_value": [1.0]},
            "MIT_weight": {"_type": "choice", "_value": [1.0]},
            "batch_size": {"_type": "choice", "_value": [8, 16, 32]},
            "lr": {"_type": "loguniform", "_value": [0.0001, 0.01]},
        },
        'ItalyAir': {
            "n_steps": {"_type": "choice", "_value": [12]},
            "n_features": {"_type": "choice", "_value": [13]},
            "epochs": {"_type": "choice", "_value": [1000]},
            "patience": {"_type": "choice", "_value": [10]},
            "pe_dim": {"_type": "choice", "_value": [3, 4, 6]},
            "feature_embed_dim": {"_type": "choice", "_value": [4, 6, 8]},
            "d_model": {"_type": "choice", "_value": [32, 40, 64]},
            "n_heads": {"_type": "choice", "_value": [2, 4, 8]},
            "n_layers": {"_type": "choice", "_value": [1, 2, 3]},
            "dropout": {"_type": "choice", "_value": [0, 0.1, 0.2]},
            "ORT_weight": {"_type": "choice", "_value": [1.0]},
            "MIT_weight": {"_type": "choice", "_value": [1.0]},
            "batch_size": {"_type": "choice", "_value": [8, 16, 32]},
            "lr": {"_type": "loguniform", "_value": [0.0001, 0.01]},
        },
    },
    'HELIX_NoSinusoidalPE': {
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
        },
        'BeijingAir': {
            "n_steps": {"_type": "choice", "_value": [24]},
            "n_features": {"_type": "choice", "_value": [132]},
            "epochs": {"_type": "choice", "_value": [1000]},
            "patience": {"_type": "choice", "_value": [10]},
            "pe_dim": {"_type": "choice", "_value": [6, 12, 24]},
            "feature_embed_dim": {"_type": "choice", "_value": [16, 24, 32]},
            "d_model": {"_type": "choice", "_value": [192, 256, 384]},
            "n_heads": {"_type": "choice", "_value": [4, 8, 12]},
            "n_layers": {"_type": "choice", "_value": [1, 2, 3]},
            "dropout": {"_type": "choice", "_value": [0, 0.1, 0.2]},
            "ORT_weight": {"_type": "choice", "_value": [1.0]},
            "MIT_weight": {"_type": "choice", "_value": [1.0]},
            "batch_size": {"_type": "choice", "_value": [4, 8, 16]},
            "lr": {"_type": "loguniform", "_value": [0.0001, 0.01]},
        },
        'PhysioNet2012': {
            "n_steps": {"_type": "choice", "_value": [48]},
            "n_features": {"_type": "choice", "_value": [35]},
            "epochs": {"_type": "choice", "_value": [1000]},
            "patience": {"_type": "choice", "_value": [10]},
            "pe_dim": {"_type": "choice", "_value": [8, 12, 16]},
            "feature_embed_dim": {"_type": "choice", "_value": [8, 10, 16]},
            "d_model": {"_type": "choice", "_value": [64, 96, 128]},
            "n_heads": {"_type": "choice", "_value": [2, 4, 8]},
            "n_layers": {"_type": "choice", "_value": [1, 2, 3]},
            "dropout": {"_type": "choice", "_value": [0, 0.1, 0.2]},
            "ORT_weight": {"_type": "choice", "_value": [1.0]},
            "MIT_weight": {"_type": "choice", "_value": [1.0]},
            "batch_size": {"_type": "choice", "_value": [8, 16, 32]},
            "lr": {"_type": "loguniform", "_value": [0.0001, 0.01]},
        },
        'ItalyAir': {
            "n_steps": {"_type": "choice", "_value": [12]},
            "n_features": {"_type": "choice", "_value": [13]},
            "epochs": {"_type": "choice", "_value": [1000]},
            "patience": {"_type": "choice", "_value": [10]},
            "pe_dim": {"_type": "choice", "_value": [3, 4, 6]},
            "feature_embed_dim": {"_type": "choice", "_value": [4, 6, 8]},
            "d_model": {"_type": "choice", "_value": [32, 40, 64]},
            "n_heads": {"_type": "choice", "_value": [2, 4, 8]},
            "n_layers": {"_type": "choice", "_value": [1, 2, 3]},
            "dropout": {"_type": "choice", "_value": [0, 0.1, 0.2]},
            "ORT_weight": {"_type": "choice", "_value": [1.0]},
            "MIT_weight": {"_type": "choice", "_value": [1.0]},
            "batch_size": {"_type": "choice", "_value": [8, 16, 32]},
            "lr": {"_type": "loguniform", "_value": [0.0001, 0.01]},
        },
    },
    'HELIX_NoFeatureEmbed': {
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
        },
        'BeijingAir': {
            "n_steps": {"_type": "choice", "_value": [24]},
            "n_features": {"_type": "choice", "_value": [132]},
            "epochs": {"_type": "choice", "_value": [1000]},
            "patience": {"_type": "choice", "_value": [10]},
            "pe_dim": {"_type": "choice", "_value": [6, 12, 24]},
            "feature_embed_dim": {"_type": "choice", "_value": [16, 24, 32]},
            "d_model": {"_type": "choice", "_value": [192, 256, 384]},
            "n_heads": {"_type": "choice", "_value": [4, 8, 12]},
            "n_layers": {"_type": "choice", "_value": [1, 2, 3]},
            "dropout": {"_type": "choice", "_value": [0, 0.1, 0.2]},
            "ORT_weight": {"_type": "choice", "_value": [1.0]},
            "MIT_weight": {"_type": "choice", "_value": [1.0]},
            "batch_size": {"_type": "choice", "_value": [4, 8, 16]},
            "lr": {"_type": "loguniform", "_value": [0.0001, 0.01]},
        },
        'PhysioNet2012': {
            "n_steps": {"_type": "choice", "_value": [48]},
            "n_features": {"_type": "choice", "_value": [35]},
            "epochs": {"_type": "choice", "_value": [1000]},
            "patience": {"_type": "choice", "_value": [10]},
            "pe_dim": {"_type": "choice", "_value": [8, 12, 16]},
            "feature_embed_dim": {"_type": "choice", "_value": [8, 10, 16]},
            "d_model": {"_type": "choice", "_value": [64, 96, 128]},
            "n_heads": {"_type": "choice", "_value": [2, 4, 8]},
            "n_layers": {"_type": "choice", "_value": [1, 2, 3]},
            "dropout": {"_type": "choice", "_value": [0, 0.1, 0.2]},
            "ORT_weight": {"_type": "choice", "_value": [1.0]},
            "MIT_weight": {"_type": "choice", "_value": [1.0]},
            "batch_size": {"_type": "choice", "_value": [8, 16, 32]},
            "lr": {"_type": "loguniform", "_value": [0.0001, 0.01]},
        },
        'ItalyAir': {
            "n_steps": {"_type": "choice", "_value": [12]},
            "n_features": {"_type": "choice", "_value": [13]},
            "epochs": {"_type": "choice", "_value": [1000]},
            "patience": {"_type": "choice", "_value": [10]},
            "pe_dim": {"_type": "choice", "_value": [3, 4, 6]},
            "feature_embed_dim": {"_type": "choice", "_value": [4, 6, 8]},
            "d_model": {"_type": "choice", "_value": [32, 40, 64]},
            "n_heads": {"_type": "choice", "_value": [2, 4, 8]},
            "n_layers": {"_type": "choice", "_value": [1, 2, 3]},
            "dropout": {"_type": "choice", "_value": [0, 0.1, 0.2]},
            "ORT_weight": {"_type": "choice", "_value": [1.0]},
            "MIT_weight": {"_type": "choice", "_value": [1.0]},
            "batch_size": {"_type": "choice", "_value": [8, 16, 32]},
            "lr": {"_type": "loguniform", "_value": [0.0001, 0.01]},
        },
    },
    'HELIX_NoHybrid': {
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
        },
        'BeijingAir': {
            "n_steps": {"_type": "choice", "_value": [24]},
            "n_features": {"_type": "choice", "_value": [132]},
            "epochs": {"_type": "choice", "_value": [1000]},
            "patience": {"_type": "choice", "_value": [10]},
            "pe_dim": {"_type": "choice", "_value": [6, 12, 24]},
            "feature_embed_dim": {"_type": "choice", "_value": [16, 24, 32]},
            "d_model": {"_type": "choice", "_value": [192, 256, 384]},
            "n_heads": {"_type": "choice", "_value": [4, 8, 12]},
            "n_layers": {"_type": "choice", "_value": [1, 2, 3]},
            "dropout": {"_type": "choice", "_value": [0, 0.1, 0.2]},
            "ORT_weight": {"_type": "choice", "_value": [1.0]},
            "MIT_weight": {"_type": "choice", "_value": [1.0]},
            "batch_size": {"_type": "choice", "_value": [4, 8, 16]},
            "lr": {"_type": "loguniform", "_value": [0.0001, 0.01]},
        },
        'PhysioNet2012': {
            "n_steps": {"_type": "choice", "_value": [48]},
            "n_features": {"_type": "choice", "_value": [35]},
            "epochs": {"_type": "choice", "_value": [1000]},
            "patience": {"_type": "choice", "_value": [10]},
            "pe_dim": {"_type": "choice", "_value": [8, 12, 16]},
            "feature_embed_dim": {"_type": "choice", "_value": [8, 10, 16]},
            "d_model": {"_type": "choice", "_value": [64, 96, 128]},
            "n_heads": {"_type": "choice", "_value": [2, 4, 8]},
            "n_layers": {"_type": "choice", "_value": [1, 2, 3]},
            "dropout": {"_type": "choice", "_value": [0, 0.1, 0.2]},
            "ORT_weight": {"_type": "choice", "_value": [1.0]},
            "MIT_weight": {"_type": "choice", "_value": [1.0]},
            "batch_size": {"_type": "choice", "_value": [8, 16, 32]},
            "lr": {"_type": "loguniform", "_value": [0.0001, 0.01]},
        },
        'ItalyAir': {
            "n_steps": {"_type": "choice", "_value": [12]},
            "n_features": {"_type": "choice", "_value": [13]},
            "epochs": {"_type": "choice", "_value": [1000]},
            "patience": {"_type": "choice", "_value": [10]},
            "pe_dim": {"_type": "choice", "_value": [3, 4, 6]},
            "feature_embed_dim": {"_type": "choice", "_value": [4, 6, 8]},
            "d_model": {"_type": "choice", "_value": [32, 40, 64]},
            "n_heads": {"_type": "choice", "_value": [2, 4, 8]},
            "n_layers": {"_type": "choice", "_value": [1, 2, 3]},
            "dropout": {"_type": "choice", "_value": [0, 0.1, 0.2]},
            "ORT_weight": {"_type": "choice", "_value": [1.0]},
            "MIT_weight": {"_type": "choice", "_value": [1.0]},
            "batch_size": {"_type": "choice", "_value": [8, 16, 32]},
            "lr": {"_type": "loguniform", "_value": [0.0001, 0.01]},
        },
    },
    'HELIX_NoFusion': {
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
        },
        'BeijingAir': {
            "n_steps": {"_type": "choice", "_value": [24]},
            "n_features": {"_type": "choice", "_value": [132]},
            "epochs": {"_type": "choice", "_value": [1000]},
            "patience": {"_type": "choice", "_value": [10]},
            "pe_dim": {"_type": "choice", "_value": [6, 12, 24]},
            "feature_embed_dim": {"_type": "choice", "_value": [16, 24, 32]},
            "d_model": {"_type": "choice", "_value": [192, 256, 384]},
            "n_heads": {"_type": "choice", "_value": [4, 8, 12]},
            "n_layers": {"_type": "choice", "_value": [1, 2, 3]},
            "dropout": {"_type": "choice", "_value": [0, 0.1, 0.2]},
            "ORT_weight": {"_type": "choice", "_value": [1.0]},
            "MIT_weight": {"_type": "choice", "_value": [1.0]},
            "batch_size": {"_type": "choice", "_value": [4, 8, 16]},
            "lr": {"_type": "loguniform", "_value": [0.0001, 0.01]},
        },
        'PhysioNet2012': {
            "n_steps": {"_type": "choice", "_value": [48]},
            "n_features": {"_type": "choice", "_value": [35]},
            "epochs": {"_type": "choice", "_value": [1000]},
            "patience": {"_type": "choice", "_value": [10]},
            "pe_dim": {"_type": "choice", "_value": [8, 12, 16]},
            "feature_embed_dim": {"_type": "choice", "_value": [8, 10, 16]},
            "d_model": {"_type": "choice", "_value": [64, 96, 128]},
            "n_heads": {"_type": "choice", "_value": [2, 4, 8]},
            "n_layers": {"_type": "choice", "_value": [1, 2, 3]},
            "dropout": {"_type": "choice", "_value": [0, 0.1, 0.2]},
            "ORT_weight": {"_type": "choice", "_value": [1.0]},
            "MIT_weight": {"_type": "choice", "_value": [1.0]},
            "batch_size": {"_type": "choice", "_value": [8, 16, 32]},
            "lr": {"_type": "loguniform", "_value": [0.0001, 0.01]},
        },
        'ItalyAir': {
            "n_steps": {"_type": "choice", "_value": [12]},
            "n_features": {"_type": "choice", "_value": [13]},
            "epochs": {"_type": "choice", "_value": [1000]},
            "patience": {"_type": "choice", "_value": [10]},
            "pe_dim": {"_type": "choice", "_value": [3, 4, 6]},
            "feature_embed_dim": {"_type": "choice", "_value": [4, 6, 8]},
            "d_model": {"_type": "choice", "_value": [32, 40, 64]},
            "n_heads": {"_type": "choice", "_value": [2, 4, 8]},
            "n_layers": {"_type": "choice", "_value": [1, 2, 3]},
            "dropout": {"_type": "choice", "_value": [0, 0.1, 0.2]},
            "ORT_weight": {"_type": "choice", "_value": [1.0]},
            "MIT_weight": {"_type": "choice", "_value": [1.0]},
            "batch_size": {"_type": "choice", "_value": [8, 16, 32]},
            "lr": {"_type": "loguniform", "_value": [0.0001, 0.01]},
        },
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
        },
        'BeijingAir': {
            "n_steps": {"_type": "choice", "_value": [24]},
            "n_features": {"_type": "choice", "_value": [132]},
            "epochs": {"_type": "choice", "_value": [1000]},
            "patience": {"_type": "choice", "_value": [10]},
            "n_fod": {"_type": "choice", "_value": [1, 2, 3]},
            "apply_nonstationary_norm": {"_type": "choice", "_value": [True, False]},
            "ORT_weight": {"_type": "choice", "_value": [1.0]},
            "MIT_weight": {"_type": "choice", "_value": [1.0]},
            "batch_size": {"_type": "choice", "_value": [4, 8, 16]},
            "lr": {"_type": "loguniform", "_value": [0.0001, 0.01]},
        },
        'PhysioNet2012': {
            "n_steps": {"_type": "choice", "_value": [48]},
            "n_features": {"_type": "choice", "_value": [35]},
            "epochs": {"_type": "choice", "_value": [1000]},
            "patience": {"_type": "choice", "_value": [10]},
            "n_fod": {"_type": "choice", "_value": [1, 2, 3]},
            "apply_nonstationary_norm": {"_type": "choice", "_value": [True, False]},
            "ORT_weight": {"_type": "choice", "_value": [1.0]},
            "MIT_weight": {"_type": "choice", "_value": [1.0]},
            "batch_size": {"_type": "choice", "_value": [8, 16, 32]},
            "lr": {"_type": "loguniform", "_value": [0.0001, 0.01]},
        },
        'ItalyAir': {
            "n_steps": {"_type": "choice", "_value": [12]},
            "n_features": {"_type": "choice", "_value": [13]},
            "epochs": {"_type": "choice", "_value": [1000]},
            "patience": {"_type": "choice", "_value": [10]},
            "n_fod": {"_type": "choice", "_value": [1, 2, 3]},
            "apply_nonstationary_norm": {"_type": "choice", "_value": [True, False]},
            "ORT_weight": {"_type": "choice", "_value": [1.0]},
            "MIT_weight": {"_type": "choice", "_value": [1.0]},
            "batch_size": {"_type": "choice", "_value": [8, 16, 32]},
            "lr": {"_type": "loguniform", "_value": [0.0001, 0.01]},
        },
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
        },
        'BeijingAir': {
            "n_steps": {"_type": "choice", "_value": [24]},
            "n_features": {"_type": "choice", "_value": [132]},
            "epochs": {"_type": "choice", "_value": [1000]},
            "patience": {"_type": "choice", "_value": [10]},
            "n_layers": {"_type": "choice", "_value": [1, 2, 3]},
            "d_model": {"_type": "choice", "_value": [64, 128, 192]},
            "d_ffn": {"_type": "choice", "_value": [128, 256, 384]},
            "top_k": {"_type": "choice", "_value": [3, 5, 7]},
            "dropout": {"_type": "choice", "_value": [0, 0.1, 0.2]},
            "channel_independence": {"_type": "choice", "_value": [True, False]},
            "decomp_method": {"_type": "choice", "_value": ["moving_avg"]},
            "moving_avg": {"_type": "choice", "_value": [3, 5, 7]},
            "downsampling_layers": {"_type": "choice", "_value": [1, 2]},
            "downsampling_window": {"_type": "choice", "_value": [2, 3, 4]},
            "apply_nonstationary_norm": {"_type": "choice", "_value": [False]},
            "batch_size": {"_type": "choice", "_value": [4, 8, 16]},
            "lr": {"_type": "loguniform", "_value": [0.0001, 0.01]},
        },
        'PhysioNet2012': {
            "n_steps": {"_type": "choice", "_value": [48]},
            "n_features": {"_type": "choice", "_value": [35]},
            "epochs": {"_type": "choice", "_value": [1000]},
            "patience": {"_type": "choice", "_value": [10]},
            "n_layers": {"_type": "choice", "_value": [1, 2, 3]},
            "d_model": {"_type": "choice", "_value": [32, 64, 128]},
            "d_ffn": {"_type": "choice", "_value": [64, 128, 256]},
            "top_k": {"_type": "choice", "_value": [3, 5, 7]},
            "dropout": {"_type": "choice", "_value": [0, 0.1, 0.2]},
            "channel_independence": {"_type": "choice", "_value": [True, False]},
            "decomp_method": {"_type": "choice", "_value": ["moving_avg"]},
            "moving_avg": {"_type": "choice", "_value": [3, 5, 7]},
            "downsampling_layers": {"_type": "choice", "_value": [1, 2]},
            "downsampling_window": {"_type": "choice", "_value": [2, 3, 4]},
            "apply_nonstationary_norm": {"_type": "choice", "_value": [False]},
            "batch_size": {"_type": "choice", "_value": [8, 16, 32]},
            "lr": {"_type": "loguniform", "_value": [0.0001, 0.01]},
        },
        'ItalyAir': {
            "n_steps": {"_type": "choice", "_value": [12]},
            "n_features": {"_type": "choice", "_value": [13]},
            "epochs": {"_type": "choice", "_value": [1000]},
            "patience": {"_type": "choice", "_value": [10]},
            "n_layers": {"_type": "choice", "_value": [1, 2, 3]},
            "d_model": {"_type": "choice", "_value": [24, 32, 48]},
            "d_ffn": {"_type": "choice", "_value": [48, 64, 96]},
            "top_k": {"_type": "choice", "_value": [2, 3, 5]},
            "dropout": {"_type": "choice", "_value": [0, 0.1, 0.2]},
            "channel_independence": {"_type": "choice", "_value": [True, False]},
            "decomp_method": {"_type": "choice", "_value": ["moving_avg"]},
            "moving_avg": {"_type": "choice", "_value": [3, 5]},
            "downsampling_layers": {"_type": "choice", "_value": [1, 2]},
            "downsampling_window": {"_type": "choice", "_value": [2, 3]},
            "apply_nonstationary_norm": {"_type": "choice", "_value": [False]},
            "batch_size": {"_type": "choice", "_value": [8, 16, 32]},
            "lr": {"_type": "loguniform", "_value": [0.0001, 0.01]},
        },
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
        },
        'BeijingAir': {
            "n_steps": {"_type": "choice", "_value": [24]},
            "n_features": {"_type": "choice", "_value": [132]},
            "epochs": {"_type": "choice", "_value": [1000]},
            "patience": {"_type": "choice", "_value": [10]},
            "patch_size": {"_type": "choice", "_value": [4, 6, 8]},
            "patch_stride": {"_type": "choice", "_value": [4, 6, 8]},
            "downsampling_ratio": {"_type": "choice", "_value": [2, 4]},
            "ffn_ratio": {"_type": "choice", "_value": [2, 4]},
            "num_blocks": {"_type": "choice", "_value": [[1, 1], [1, 2]]},
            "large_size": {"_type": "choice", "_value": [[7, 7], [5, 5]]},
            "small_size": {"_type": "choice", "_value": [[3, 3]]},
            "dims": {"_type": "choice", "_value": [[48, 48], [64, 64], [48, 64]]},
            "small_kernel_merged": {"_type": "choice", "_value": [False]},
            "backbone_dropout": {"_type": "choice", "_value": [0, 0.1, 0.2]},
            "head_dropout": {"_type": "choice", "_value": [0, 0.1, 0.2]},
            "use_multi_scale": {"_type": "choice", "_value": [False]},
            "individual": {"_type": "choice", "_value": [False]},
            "apply_nonstationary_norm": {"_type": "choice", "_value": [False]},
            "batch_size": {"_type": "choice", "_value": [4, 8, 16]},
            "lr": {"_type": "loguniform", "_value": [0.0001, 0.01]},
        },
        'PhysioNet2012': {
            "n_steps": {"_type": "choice", "_value": [48]},
            "n_features": {"_type": "choice", "_value": [35]},
            "epochs": {"_type": "choice", "_value": [1000]},
            "patience": {"_type": "choice", "_value": [10]},
            "patch_size": {"_type": "choice", "_value": [6, 8, 12]},
            "patch_stride": {"_type": "choice", "_value": [6, 8, 12]},
            "downsampling_ratio": {"_type": "choice", "_value": [2, 4]},
            "ffn_ratio": {"_type": "choice", "_value": [2, 4]},
            "num_blocks": {"_type": "choice", "_value": [[1, 1], [1, 2]]},
            "large_size": {"_type": "choice", "_value": [[7, 7], [5, 5]]},
            "small_size": {"_type": "choice", "_value": [[3, 3]]},
            "dims": {"_type": "choice", "_value": [[24, 24], [32, 32], [24, 32]]},
            "small_kernel_merged": {"_type": "choice", "_value": [False]},
            "backbone_dropout": {"_type": "choice", "_value": [0, 0.1, 0.2]},
            "head_dropout": {"_type": "choice", "_value": [0, 0.1, 0.2]},
            "use_multi_scale": {"_type": "choice", "_value": [False]},
            "individual": {"_type": "choice", "_value": [False]},
            "apply_nonstationary_norm": {"_type": "choice", "_value": [False]},
            "batch_size": {"_type": "choice", "_value": [8, 16, 32]},
            "lr": {"_type": "loguniform", "_value": [0.0001, 0.01]},
        },
        'ItalyAir': {
            "n_steps": {"_type": "choice", "_value": [12]},
            "n_features": {"_type": "choice", "_value": [13]},
            "epochs": {"_type": "choice", "_value": [1000]},
            "patience": {"_type": "choice", "_value": [10]},
            "patch_size": {"_type": "choice", "_value": [3, 4, 6]},
            "patch_stride": {"_type": "choice", "_value": [3, 4, 6]},
            "downsampling_ratio": {"_type": "choice", "_value": [2, 3]},
            "ffn_ratio": {"_type": "choice", "_value": [2, 4]},
            "num_blocks": {"_type": "choice", "_value": [[1], [1, 1]]},
            "large_size": {"_type": "choice", "_value": [[5], [5, 5]]},
            "small_size": {"_type": "choice", "_value": [[3], [3, 3]]},
            "dims": {"_type": "choice", "_value": [[24], [32], [24, 32]]},
            "small_kernel_merged": {"_type": "choice", "_value": [False]},
            "backbone_dropout": {"_type": "choice", "_value": [0, 0.1, 0.2]},
            "head_dropout": {"_type": "choice", "_value": [0, 0.1, 0.2]},
            "use_multi_scale": {"_type": "choice", "_value": [False]},
            "individual": {"_type": "choice", "_value": [False]},
            "apply_nonstationary_norm": {"_type": "choice", "_value": [False]},
            "batch_size": {"_type": "choice", "_value": [8, 16, 32]},
            "lr": {"_type": "loguniform", "_value": [0.0001, 0.01]},
        },
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
        },
        'BeijingAir': {
            "n_steps": {"_type": "choice", "_value": [24]},
            "n_features": {"_type": "choice", "_value": [132]},
            "epochs": {"_type": "choice", "_value": [1000]},
            "patience": {"_type": "choice", "_value": [10]},
            "n_layers": {"_type": "choice", "_value": [1, 2, 3]},
            "d_input_embed": {"_type": "choice", "_value": [48, 64, 96]},
            "d_learnable_embed": {"_type": "choice", "_value": [48, 64, 96]},
            "d_proj": {"_type": "choice", "_value": [48, 64, 96]},
            "d_ffn": {"_type": "choice", "_value": [96, 128, 192]},
            "n_temporal_heads": {"_type": "choice", "_value": [2, 4, 8]},
            "dropout": {"_type": "choice", "_value": [0, 0.1, 0.2]},
            "input_dim": {"_type": "choice", "_value": [1]},
            "output_dim": {"_type": "choice", "_value": [1]},
            "ORT_weight": {"_type": "choice", "_value": [1.0]},
            "MIT_weight": {"_type": "choice", "_value": [1.0]},
            "batch_size": {"_type": "choice", "_value": [4, 8, 16]},
            "lr": {"_type": "loguniform", "_value": [0.0001, 0.01]},
        },
        'PhysioNet2012': {
            "n_steps": {"_type": "choice", "_value": [48]},
            "n_features": {"_type": "choice", "_value": [35]},
            "epochs": {"_type": "choice", "_value": [1000]},
            "patience": {"_type": "choice", "_value": [10]},
            "n_layers": {"_type": "choice", "_value": [1, 2, 3]},
            "d_input_embed": {"_type": "choice", "_value": [32, 48, 64]},
            "d_learnable_embed": {"_type": "choice", "_value": [32, 48, 64]},
            "d_proj": {"_type": "choice", "_value": [32, 48, 64]},
            "d_ffn": {"_type": "choice", "_value": [64, 96, 128]},
            "n_temporal_heads": {"_type": "choice", "_value": [2, 4, 8]},
            "dropout": {"_type": "choice", "_value": [0, 0.1, 0.2]},
            "input_dim": {"_type": "choice", "_value": [1]},
            "output_dim": {"_type": "choice", "_value": [1]},
            "ORT_weight": {"_type": "choice", "_value": [1.0]},
            "MIT_weight": {"_type": "choice", "_value": [1.0]},
            "batch_size": {"_type": "choice", "_value": [8, 16, 32]},
            "lr": {"_type": "loguniform", "_value": [0.0001, 0.01]},
        },
        'ItalyAir': {
            "n_steps": {"_type": "choice", "_value": [12]},
            "n_features": {"_type": "choice", "_value": [13]},
            "epochs": {"_type": "choice", "_value": [1000]},
            "patience": {"_type": "choice", "_value": [10]},
            "n_layers": {"_type": "choice", "_value": [1, 2, 3]},
            "d_input_embed": {"_type": "choice", "_value": [16, 24, 32]},
            "d_learnable_embed": {"_type": "choice", "_value": [16, 24, 32]},
            "d_proj": {"_type": "choice", "_value": [16, 24, 32]},
            "d_ffn": {"_type": "choice", "_value": [32, 48, 64]},
            "n_temporal_heads": {"_type": "choice", "_value": [2, 4, 8]},
            "dropout": {"_type": "choice", "_value": [0, 0.1, 0.2]},
            "input_dim": {"_type": "choice", "_value": [1]},
            "output_dim": {"_type": "choice", "_value": [1]},
            "ORT_weight": {"_type": "choice", "_value": [1.0]},
            "MIT_weight": {"_type": "choice", "_value": [1.0]},
            "batch_size": {"_type": "choice", "_value": [8, 16, 32]},
            "lr": {"_type": "loguniform", "_value": [0.0001, 0.01]},
        },
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
        },
        'BeijingAir': {
            "n_steps": {"_type": "choice", "_value": [24]},
            "n_features": {"_type": "choice", "_value": [132]},
            "epochs": {"_type": "choice", "_value": [1000]},
            "patience": {"_type": "choice", "_value": [10]},
            "d_block_hidden": {"_type": "choice", "_value": [48, 64, 96]},
            "n_residual_layers": {"_type": "choice", "_value": [1, 2, 3]},
            "d_residual_hidden": {"_type": "choice", "_value": [24, 32, 48]},
            "d_embedding": {"_type": "choice", "_value": [48, 64, 96]},
            "n_embeddings": {"_type": "choice", "_value": [256, 512, 768]},
            "commitment_cost": {"_type": "choice", "_value": [0.1, 0.25, 0.5]},
            "compression_factor": {"_type": "choice", "_value": [2, 3, 4]},
            "batch_size": {"_type": "choice", "_value": [4, 8, 16]},
            "lr": {"_type": "loguniform", "_value": [0.0001, 0.01]},
        },
        'PhysioNet2012': {
            "n_steps": {"_type": "choice", "_value": [48]},
            "n_features": {"_type": "choice", "_value": [35]},
            "epochs": {"_type": "choice", "_value": [1000]},
            "patience": {"_type": "choice", "_value": [10]},
            "d_block_hidden": {"_type": "choice", "_value": [32, 48, 64]},
            "n_residual_layers": {"_type": "choice", "_value": [1, 2, 3]},
            "d_residual_hidden": {"_type": "choice", "_value": [16, 24, 32]},
            "d_embedding": {"_type": "choice", "_value": [32, 48, 64]},
            "n_embeddings": {"_type": "choice", "_value": [128, 256, 512]},
            "commitment_cost": {"_type": "choice", "_value": [0.1, 0.25, 0.5]},
            "compression_factor": {"_type": "choice", "_value": [2, 3, 4]},
            "batch_size": {"_type": "choice", "_value": [8, 16, 32]},
            "lr": {"_type": "loguniform", "_value": [0.0001, 0.01]},
        },
        'ItalyAir': {
            "n_steps": {"_type": "choice", "_value": [12]},
            "n_features": {"_type": "choice", "_value": [13]},
            "epochs": {"_type": "choice", "_value": [1000]},
            "patience": {"_type": "choice", "_value": [10]},
            "d_block_hidden": {"_type": "choice", "_value": [16, 24, 32]},
            "n_residual_layers": {"_type": "choice", "_value": [1, 2, 3]},
            "d_residual_hidden": {"_type": "choice", "_value": [8, 12, 16]},
            "d_embedding": {"_type": "choice", "_value": [16, 24, 32]},
            "n_embeddings": {"_type": "choice", "_value": [64, 128, 256]},
            "commitment_cost": {"_type": "choice", "_value": [0.1, 0.25, 0.5]},
            "compression_factor": {"_type": "choice", "_value": [2, 3, 4]},
            "batch_size": {"_type": "choice", "_value": [8, 16, 32]},
            "lr": {"_type": "loguniform", "_value": [0.00005, 0.001]},
        },
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
        },
        'BeijingAir': {
            "n_steps": {"_type": "choice", "_value": [24]},
            "n_features": {"_type": "choice", "_value": [132]},
            "epochs": {"_type": "choice", "_value": [1000]},
            "patience": {"_type": "choice", "_value": [10]},
            "n_layers": {"_type": "choice", "_value": [1, 2, 3]},
            "d_model": {"_type": "choice", "_value": [64, 128, 192]},
            "d_ffn": {"_type": "choice", "_value": [128, 256, 384]},
            "top_k": {"_type": "choice", "_value": [3, 5, 7]},
            "n_heads": {"_type": "choice", "_value": [2, 4, 8]},
            "n_kernels": {"_type": "choice", "_value": [4, 6, 8]},
            "dropout": {"_type": "choice", "_value": [0, 0.1, 0.2]},
            "channel_mixing": {"_type": "choice", "_value": [True, False]},
            "channel_independence": {"_type": "choice", "_value": [True, False]},
            "downsampling_layers": {"_type": "choice", "_value": [1, 2]},
            "downsampling_window": {"_type": "choice", "_value": [2, 3, 4]},
            "apply_nonstationary_norm": {"_type": "choice", "_value": [False]},
            "batch_size": {"_type": "choice", "_value": [4, 8, 16]},
            "lr": {"_type": "loguniform", "_value": [0.0001, 0.01]},
        },
        'PhysioNet2012': {
            "n_steps": {"_type": "choice", "_value": [48]},
            "n_features": {"_type": "choice", "_value": [35]},
            "epochs": {"_type": "choice", "_value": [1000]},
            "patience": {"_type": "choice", "_value": [10]},
            "n_layers": {"_type": "choice", "_value": [1, 2, 3]},
            "d_model": {"_type": "choice", "_value": [32, 64, 128]},
            "d_ffn": {"_type": "choice", "_value": [64, 128, 256]},
            "top_k": {"_type": "choice", "_value": [2, 3, 5]},
            "n_heads": {"_type": "choice", "_value": [2, 4, 8]},
            "n_kernels": {"_type": "choice", "_value": [4, 6, 8]},
            "dropout": {"_type": "choice", "_value": [0, 0.1, 0.2]},
            "channel_mixing": {"_type": "choice", "_value": [True, False]},
            "channel_independence": {"_type": "choice", "_value": [True, False]},
            "downsampling_layers": {"_type": "choice", "_value": [1, 2]},
            "downsampling_window": {"_type": "choice", "_value": [2, 3, 4]},
            "apply_nonstationary_norm": {"_type": "choice", "_value": [False]},
            "batch_size": {"_type": "choice", "_value": [8, 16, 32]},
            "lr": {"_type": "loguniform", "_value": [0.0001, 0.01]},
        },
        'ItalyAir': {
            "n_steps": {"_type": "choice", "_value": [12]},
            "n_features": {"_type": "choice", "_value": [13]},
            "epochs": {"_type": "choice", "_value": [1000]},
            "patience": {"_type": "choice", "_value": [10]},
            "n_layers": {"_type": "choice", "_value": [1, 2, 3]},
            "d_model": {"_type": "choice", "_value": [24, 32, 48]},
            "d_ffn": {"_type": "choice", "_value": [48, 64, 96]},
            "top_k": {"_type": "choice", "_value": [2, 3, 5]},
            "n_heads": {"_type": "choice", "_value": [2, 4, 8]},
            "n_kernels": {"_type": "choice", "_value": [4, 6, 8]},
            "dropout": {"_type": "choice", "_value": [0, 0.1, 0.2]},
            "channel_mixing": {"_type": "choice", "_value": [True, False]},
            "channel_independence": {"_type": "choice", "_value": [True, False]},
            "downsampling_layers": {"_type": "choice", "_value": [1, 2]},
            "downsampling_window": {"_type": "choice", "_value": [2, 3]},
            "apply_nonstationary_norm": {"_type": "choice", "_value": [False]},
            "batch_size": {"_type": "choice", "_value": [8, 16, 32]},
            "lr": {"_type": "loguniform", "_value": [0.0001, 0.01]},
        },
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
        },
        'BeijingAir': {
            "n_steps": {"_type": "choice", "_value": [24]},
            "n_features": {"_type": "choice", "_value": [132]},
            "epochs": {"_type": "choice", "_value": [1000]},
            "patience": {"_type": "choice", "_value": [10]},
            "patch_size": {"_type": "choice", "_value": [6, 8, 12]},
            "patch_stride": {"_type": "choice", "_value": [6, 8, 12]},
            "transformer_backbone": {"_type": "choice", "_value": ["t5-base"]},
            "transformer_type": {"_type": "choice", "_value": ["encoder_only"]},
            "d_model": {"_type": "choice", "_value": [768]},
            "revin_affine": {"_type": "choice", "_value": [True]},
            "add_positional_embedding": {"_type": "choice", "_value": [True]},
            "value_embedding_bias": {"_type": "choice", "_value": [True]},
            "orth_gain": {"_type": "choice", "_value": [1.41]},
            "n_layers": {"_type": "choice", "_value": [2, 4, 6]},
            "d_ffn": {"_type": "choice", "_value": [1024, 2048, 4096]},
            "dropout": {"_type": "choice", "_value": [0, 0.1, 0.2]},
            "head_dropout": {"_type": "choice", "_value": [0, 0.1, 0.2]},
            "finetuning_mode": {"_type": "choice", "_value": ["linear-probing", "end-to-end"]},
            "mask_ratio": {"_type": "choice", "_value": [0.1, 0.3, 0.5]},
            "batch_size": {"_type": "choice", "_value": [2, 4, 8]},
            "lr": {"_type": "loguniform", "_value": [0.00005, 0.001]},
        },
        'PhysioNet2012': {
            "n_steps": {"_type": "choice", "_value": [48]},
            "n_features": {"_type": "choice", "_value": [35]},
            "epochs": {"_type": "choice", "_value": [100]},
            "patience": {"_type": "choice", "_value": [5]},
            "patch_size": {"_type": "choice", "_value": [12, 24]},
            "patch_stride": {"_type": "choice", "_value": [12, 24]},
            "transformer_backbone": {"_type": "choice", "_value": ["t5-small"]},
            "transformer_type": {"_type": "choice", "_value": ["encoder_only"]},
            "d_model": {"_type": "choice", "_value": [512]},
            "revin_affine": {"_type": "choice", "_value": [True]},
            "add_positional_embedding": {"_type": "choice", "_value": [True]},
            "value_embedding_bias": {"_type": "choice", "_value": [True]},
            "orth_gain": {"_type": "choice", "_value": [1.41]},
            "n_layers": {"_type": "choice", "_value": [2, 4]},
            "d_ffn": {"_type": "choice", "_value": [512, 1024, 2048]},
            "dropout": {"_type": "choice", "_value": [0, 0.1, 0.2]},
            "head_dropout": {"_type": "choice", "_value": [0, 0.1, 0.2]},
            "finetuning_mode": {"_type": "choice", "_value": ["linear-probing", "end-to-end"]},
            "mask_ratio": {"_type": "choice", "_value": [0.1, 0.3, 0.5]},
            "batch_size": {"_type": "choice", "_value": [8, 16, 32]},
            "lr": {"_type": "loguniform", "_value": [0.0005, 0.005]},
        }
    },
}


def format_value(param_config):
    """格式化参数值为LaTeX字符串"""
    param_type = param_config["_type"]
    values = param_config["_value"]
    
    if param_type == "loguniform":
        # 科学计数法格式化
        low, high = values
        if low < 0.001:
            low_str = f"{low:.0e}".replace("e-0", "e-").replace("e+0", "e+")
        else:
            low_str = str(low)
        if high < 0.001:
            high_str = f"{high:.0e}".replace("e-0", "e-").replace("e+0", "e+")
        else:
            high_str = str(high)
        return f"[{low_str}, {high_str}] (log)"
    
    elif param_type == "choice":
        if len(values) == 1:
            # 单一值
            val = values[0]
            if isinstance(val, bool):
                return "True" if val else "False"
            elif isinstance(val, list):
                return str(val)
            elif isinstance(val, str):
                if len(val) > 20:
                    return "(see note)"
                return f'"{val}"'
            else:
                return str(val)
        else:
            # 多个选项
            formatted = []
            for v in values:
                if isinstance(v, bool):
                    formatted.append("T" if v else "F")
                elif isinstance(v, list):
                    formatted.append(str(v))
                elif isinstance(v, str):
                    # 缩短长字符串
                    if len(v) > 20:
                        formatted.append(f'(see note)')
                    else:
                        formatted.append(f'"{v}"')
                else:
                    formatted.append(str(v))
            return "\\{" + ", ".join(formatted) + "\\}"
    
    return str(values)


def escape_latex(text):
    """转义LaTeX特殊字符"""
    replacements = {
        '_': '\\_',
        '%': '\\%',
        '&': '\\&',
        '#': '\\#',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def get_all_params_for_model(model_config):
    """获取模型在所有数据集上的所有参数"""
    all_params = set()
    for dataset in model_config.values():
        all_params.update(dataset.keys())
    return all_params


def categorize_params(all_params):
    """将参数分类到不同组"""
    dataset_props = [p for p in DATASET_PROPERTIES if p in all_params]
    training = [p for p in TRAINING_CONFIG if p in all_params]
    loss = [p for p in LOSS_WEIGHTS if p in all_params]
    
    # 其余归入架构参数
    categorized = set(dataset_props + training + loss)
    architecture = sorted([p for p in all_params if p not in categorized])
    
    return {
        'Dataset Properties': dataset_props,
        'Model Architecture': architecture,
        'Training Configuration': training,
        'Loss Weights': loss
    }


def generate_table_for_model(model_name, model_config, is_helix=False):
    """为单个模型生成LaTeX表格"""
    
    # 获取所有参数并分类
    all_params = get_all_params_for_model(model_config)
    categorized = categorize_params(all_params)
    
    # 确定哪些数据集有配置
    available_datasets = [d for d in DATASET_ORDER if d in model_config]
    missing_datasets = [d for d in DATASET_ORDER if d not in model_config]
    
    # 构建表格
    num_cols = len(DATASET_ORDER) + 1  # Parameter + datasets
    col_spec = "l|" + "c" * len(DATASET_ORDER)
    
    lines = []
    
    # 表格标题和脚注
    if is_helix:
        caption = f"{model_name} Hyperparameter Search Space"
        footnote = "The ablation variants (w/o Sinusoidal PE, w/o Feature Identity Embedding, w/o Hybrid Encoding, w/o Multi-level Fusion) use the same search space."
    elif model_name == 'TimeLLM':
        caption = f"{model_name} Hyperparameter Search Space"
        footnote = "Time-LLM was only evaluated on ETT-h1 due to its computational requirements. The domain\\_prompt\\_content was set to ``Electricity transformer temperature time series data''."
    elif model_name == 'MOMENT':
        caption = f"{model_name} Hyperparameter Search Space"
        footnote = "MOMENT was not evaluated on ItalyAir and PeMS due to sequence length constraints (requires minimum patch count)."
    elif model_name == 'TimeMixerPP':
        caption = f"TimeMixer++ Hyperparameter Search Space"
        footnote = "TimeMixer++ was not evaluated on PeMS due to excessive memory requirements."
    else:
        caption = f"{model_name} Hyperparameter Search Space"
        footnote = None
    
    # 开始表格
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{tab:hyperparam_{model_name.lower().replace('+', 'pp')}}}")
    lines.append("\\begin{adjustbox}{max width=\\textwidth}")
    lines.append("\\begin{small}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    
    # 表头
    header = "\\textbf{Parameter}"
    for d in DATASET_ORDER:
        header += f" & \\textbf{{{DATASET_DISPLAY[d]}}}"
    header += " \\\\"
    lines.append(header)
    lines.append("\\midrule")
    
    # 按类别输出参数
    for category, params in categorized.items():
        if not params:
            continue
        
        # 类别标题
        lines.append(f"\\multicolumn{{{num_cols}}}{{l}}{{\\textit{{{category}}}}} \\\\")
        
        for param in params:
            row = escape_latex(param)
            for dataset in DATASET_ORDER:
                if dataset in model_config and param in model_config[dataset]:
                    val = format_value(model_config[dataset][param])
                    # 转义LaTeX特殊字符（但保留我们故意添加的）
                    val = val.replace('_', '\\_')
                else:
                    val = "--"
                row += f" & {val}"
            row += " \\\\"
            lines.append(row)
        
        lines.append("\\midrule")
    
    # 移除最后一个 midrule，替换为 bottomrule
    lines[-1] = "\\bottomrule"
    
    lines.append("\\end{tabular}")
    lines.append("\\end{small}")
    lines.append("\\end{adjustbox}")
    
    if footnote:
        lines.append(f"\\\\[2pt]")
        lines.append(f"\\footnotesize{{\\textit{{Note: {footnote}}}}}")
    
    lines.append("\\end{table}")
    
    return "\n".join(lines)


def generate_all_tables():
    """生成所有表格"""
    
    all_content = []
    
    # LaTeX preamble（需要的包）
    preamble = """% Required packages (add to document preamble if not already present):
% \\usepackage{adjustbox}
% \\usepackage{booktabs}
% \\usepackage{float}

% ==============================================================================
% HYPERPARAMETER SEARCH SPACES
% ==============================================================================

"""
    all_content.append(preamble)
    
    # 生成HELIX表格（含脚注说明变体）
    helix_table = generate_table_for_model('HELIX', TUNING_SPACES['HELIX'], is_helix=True)
    all_content.append(helix_table)
    all_content.append("\n\\clearpage\n")
    
    # 保存HELIX单独的文件
    with open(os.path.join(OUTPUT_DIR, 'HELIX_hyperparams.tex'), 'w') as f:
        f.write(helix_table)
    print(f"✓ Generated: HELIX_hyperparams.tex")
    
    # 生成其他模型的表格
    for model in MODEL_ORDER[1:]:  # 跳过HELIX
        if model in TUNING_SPACES:
            table = generate_table_for_model(model, TUNING_SPACES[model])
            all_content.append(table)
            all_content.append("\n")
            
            # 保存单独文件
            filename = f"{model.replace('+', 'PP')}_hyperparams.tex"
            with open(os.path.join(OUTPUT_DIR, filename), 'w') as f:
                f.write(table)
            print(f"✓ Generated: {filename}")
    
    # 生成汇总文件
    with open(os.path.join(OUTPUT_DIR, 'all_hyperparams.tex'), 'w') as f:
        f.write("\n".join(all_content))
    print(f"\n✓ Generated: all_hyperparams.tex (complete file for direct copy-paste)")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    generate_all_tables()
    print(f"\nAll files saved to: {OUTPUT_DIR}")