"""
超参数调优专用训练脚本
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# Modified for hyperparameter tuning
# License: BSD-3-Clause

import argparse
import os
import time
import numpy as np
import torch
from pypots.data.saving import pickle_dump
from pypots.imputation import *
from pypots.optim import Adam
from pypots.utils.logging import logger
from pypots.nn.functional import calc_mae, calc_mse, calc_mre
from pypots.utils.random import set_random_seed

from global_config import TORCH_N_THREADS, RANDOM_SEEDS
from utils import get_datasets_path

SUPPORT_MODELS = {
    "Autoformer": Autoformer,
    "BRITS": BRITS,
    "Crossformer": Crossformer,
    "CSDI": CSDI,
    "DLinear": DLinear,
    "ETSformer": ETSformer,
    "FiLM": FiLM,
    "FreTS": FreTS,
    "GPVAE": GPVAE,
    "GRUD": GRUD,
    "HELIX": HELIX,
    "Informer": Informer,
    "iTransformer": iTransformer,
    "ImputeFormer": ImputeFormer,
    "Koopa": Koopa,
    "MICN": MICN,
    "MOMENT": MOMENT,
    "ModernTCN": ModernTCN,
    "MRNN": MRNN,
    "NonstationaryTransformer": NonstationaryTransformer,
    "PatchTST": PatchTST,
    "Pyraformer": Pyraformer,
    "SAITS": SAITS,
    "SCINet": SCINet,
    "StemGNN": StemGNN,
    "TEFN": TEFN,
    "TimeLLM": TimeLLM,
    "TimeMixer": TimeMixer,
    "TimeMixerPP": TimeMixerPP,
    "TimesNet": TimesNet,
    "TOTEM": TOTEM,
    "Transformer": Transformer,
    "USGAN": USGAN,
}


def parse_value(value_str):
    """解析命令行参数值"""
    # 处理布尔值
    if value_str.lower() in ['true', 'false']:
        return value_str.lower() == 'true'
    
    # 处理列表 [1,2,3]
    if value_str.startswith('[') and value_str.endswith(']'):
        content = value_str[1:-1]
        if not content:
            return []
        return [int(x) if x.isdigit() else float(x) for x in content.split(',')]
    
    # 处理数字
    try:
        if '.' in value_str:
            return float(value_str)
        else:
            return int(value_str)
    except ValueError:
        return value_str


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=list(SUPPORT_MODELS.keys()))
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--dataset_fold_path", type=str, required=True)
    parser.add_argument("--saving_path", type=str, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--n_rounds", type=int, default=1)
    
    # 解析已知参数
    args, unknown = parser.parse_known_args()
    
    # 解析超参数（所有未知参数）
    hyperparameters = {}
    i = 0
    while i < len(unknown):
        if unknown[i].startswith('--'):
            param_name = unknown[i][2:]
            if i + 1 < len(unknown) and not unknown[i + 1].startswith('--'):
                param_value = parse_value(unknown[i + 1])
                hyperparameters[param_name] = param_value
                i += 2
            else:
                i += 1
        else:
            i += 1
    
    # 打印接收到的超参数
    logger.info(f"Model: {args.model}, Dataset: {args.dataset}")
    logger.info(f"Hyperparameters: {hyperparameters}")
    
    # 设置线程数
    torch.set_num_threads(TORCH_N_THREADS)
    
    # 创建保存目录（重要！）
    result_saving_path = args.saving_path
    os.makedirs(result_saving_path, exist_ok=True)
    
    try:
        # 加载数据
        (train_set, val_set, test_X, test_X_ori, test_indicating_mask) = get_datasets_path(args.dataset_fold_path)
        
        mae_collector = []
        mse_collector = []
        mre_collector = []
        time_collector = []
        
        for n_round in range(args.n_rounds):
            set_random_seed(RANDOM_SEEDS[n_round])
            round_saving_path = os.path.join(result_saving_path, f"round_{n_round}")
            
            # 提取学习率
            lr = hyperparameters.pop("lr", 0.001)
            
            # 设置模型参数
            model_params = hyperparameters.copy()
            model_params["device"] = args.device
            model_params["saving_path"] = round_saving_path
            model_params["model_saving_strategy"] = "best"
            
            # 设置优化器
            if args.model == "USGAN":
                model_params["G_optimizer"] = Adam(lr=lr)
                model_params["D_optimizer"] = Adam(lr=lr)
            else:
                model_params["optimizer"] = Adam(lr=lr)
            
            # 创建并训练模型
            model = SUPPORT_MODELS[args.model](**model_params)
            model.fit(train_set=train_set, val_set=val_set)
            
            # 预测
            test_set = {"X": test_X}
            start_time = time.time()
            
            if args.model in ["CSDI", "GPVAE"]:
                results = model.predict(test_set, n_sampling_times=10)
                test_set_imputation = results["imputation"].mean(axis=1)
            else:
                results = model.predict(test_set)
                test_set_imputation = results["imputation"]
            
            time_collector.append(time.time() - start_time)
            
            # 计算指标
            mae = calc_mae(test_set_imputation, test_X_ori, test_indicating_mask)
            mse = calc_mse(test_set_imputation, test_X_ori, test_indicating_mask)
            mre = calc_mre(test_set_imputation, test_X_ori, test_indicating_mask)
            
            mae_collector.append(mae)
            mse_collector.append(mse)
            mre_collector.append(mre)
            
            # 保存结果
            pickle_dump(
                {"test_set_imputation": test_set_imputation},
                os.path.join(round_saving_path, "imputation.pkl"),
            )
            
            logger.info(
                f"Round{n_round} - {args.model}: MAE={mae:.4f}, MSE={mse:.4f}, MRE={mre:.4f}"
            )
        
        # 计算平均结果
        if mae_collector:
            mean_mae = np.mean(mae_collector)
            mean_mse = np.mean(mse_collector)
            mean_mre = np.mean(mre_collector)
            
            logger.info(
                f"Final results: MAE={mean_mae:.4f}, MSE={mean_mse:.4f}, MRE={mean_mre:.4f}, "
                f"avg_time={np.mean(time_collector):.2f}s"
            )
            
            # 保存最终指标
            import json
            with open(os.path.join(result_saving_path, "metrics.json"), 'w') as f:
                json.dump({
                    "mae": float(mean_mae),
                    "mse": float(mean_mse),
                    "mre": float(mean_mre),
                    "inference_time": float(np.mean(time_collector))
                }, f, indent=2)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        
        # 确保目录存在再写入错误信息
        os.makedirs(result_saving_path, exist_ok=True)
        
        # 保存失败信息
        import traceback
        with open(os.path.join(result_saving_path, "training_failed.txt"), 'w') as f:
            f.write(f"Error: {str(e)}\n\n")
            f.write(f"Full traceback:\n")
            f.write(traceback.format_exc())
        
        # 重新抛出异常以便在日志中看到
        raise