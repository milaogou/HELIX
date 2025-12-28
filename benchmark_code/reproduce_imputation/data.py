import os
import re
import pandas as pd

model_names = [
    'HELIX',
    'TEFN',
    'TimeMixerPP',
    'TimeLLM',
    'MOMENT',
    'TimeMixer',
    'ModernTCN',
    'ImputeFormer',
    'TOTEM',
    'iTransformer',
    'SAITS',
    'FreTS',
    'NonstationaryTransformer',
    'PatchTST',
]

# 定义模型的config_version映射（用于查找对应的log文件）
MODEL_CONFIG_VERSIONS = {
    # 'HELIX': 'with_LR_decay',  # HELIX的新版本后缀
}

metrics_pattern = re.compile(r"MAE=(\d+\.\d+) ± (\d+\.\d+), MSE=(\d+\.\d+) ± (\d+\.\d+), MRE=(\d+\.\d+) ± (\d+\.\d+), average inference time=(\d+\.\d+)")
params_pattern = re.compile(r"the number of trainable parameters: ([\d,]+)")

def find_log_file(log_dir, model, dataset, config_version=""):
    """
    按优先级查找日志文件：
    1. 优先查找新版本（带下划线后缀）
    2. 找不到再查找旧版本
    """
    possible_filenames = []
    
    if config_version:
        # 带config_version的情况
        # 优先新版
        possible_filenames.append(f"{model}_{dataset}_{config_version}_.log")
        # 然后旧版
        possible_filenames.append(f"{model}_{dataset}_{config_version}.log")
    else:
        # 不带config_version的情况
        # 优先新版（带下划线）
        possible_filenames.append(f"{model}_{dataset}_.log")
        # 然后旧版
        possible_filenames.append(f"{model}_{dataset}.log")
    
    # 按顺序查找，返回第一个存在的文件路径
    for filename in possible_filenames:
        file_path = os.path.join(log_dir, filename)
        if os.path.exists(file_path):
            return file_path
    
    return None

def process_dataset_logs(dataset, log_pattern):
    """处理单个数据集的日志文件"""
    log_dir = f"{log_pattern}/{dataset}_log"
    
    results = {
        "Model": [],
        "Size": [],
        "MAE": [],
        "MSE": [],
        "MRE": [],
        "Time": []
    }

    for model in model_names:
        # 获取该模型的config_version后缀
        config_version = MODEL_CONFIG_VERSIONS.get(model, "")
        
        # 查找日志文件（自动适配新旧版本）
        file_path = find_log_file(log_dir, model, dataset, config_version)
        
        if file_path:
            with open(file_path, 'r') as file:
                lines = file.readlines()
                params = "0"
                found_metrics = False
                
                for line in lines:
                    if params_pattern.search(line):
                        params_match = params_pattern.search(line)
                        params = params_match.group(1)
                    if metrics_pattern.search(line):
                        metrics_match = metrics_pattern.search(line)
                        if metrics_match:
                            mae, mae_std, mse, mse_std, mre, mre_std, time = metrics_match.groups()
                            results["Model"].append(model)
                            results["Size"].append(params)
                            results["MAE"].append(f"{float(mae):.3f} ({float(mae_std):.3f})")
                            results["MSE"].append(f"{float(mse):.3f} ({float(mse_std):.3f})")
                            results["MRE"].append(f"{float(mre):.3f} ({float(mre_std):.3f})")
                            results["Time"].append(time)
                            found_metrics = True
                            break
                
                if not found_metrics:
                    results["Model"].append(model)
                    results["Size"].append(params)
                    results["MAE"].append("0")
                    results["MSE"].append("0")
                    results["MRE"].append("0")
                    results["Time"].append("0")
        else:
            # 文件不存在
            results["Model"].append(model)
            results["Size"].append("0")
            results["MAE"].append("0")
            results["MSE"].append("0")
            results["MRE"].append("0")
            results["Time"].append("0")

    return pd.DataFrame(results)

# point01实验
for dataset in ["BeijingAir", "Electricity", "ETT_h1", "ItalyAir", "PeMS", "PhysioNet2012"]:
    df = process_dataset_logs(dataset, "point01_log")
    os.makedirs("results_csv/imputation/point01", exist_ok=True)
    df.to_csv(f"results_csv/imputation/point01/{dataset}.csv", index=False)

# point05实验
for dataset in ["BeijingAir", "Electricity", "ETT_h1", "ItalyAir", "PeMS"]:
    df = process_dataset_logs(dataset, "point05_log")
    os.makedirs("results_csv/imputation/point05", exist_ok=True)
    df.to_csv(f"results_csv/imputation/point05/{dataset}.csv", index=False)

# point09实验
for dataset in ["BeijingAir", "Electricity", "ETT_h1", "ItalyAir", "PeMS"]:
    df = process_dataset_logs(dataset, "point09_log")
    os.makedirs("results_csv/imputation/point09", exist_ok=True)
    df.to_csv(f"results_csv/imputation/point09/{dataset}.csv", index=False)

block_datasets = ["BeijingAir", "Electricity", "ETT_h1", "ItalyAir", "PeMS"]
os.makedirs("results_csv/imputation/block05", exist_ok=True)

# 处理 block00
for dataset in block_datasets:
    df = process_dataset_logs(dataset, "block00_log")
    df.to_csv(f"results_csv/imputation/block05/{dataset}.csv", index=False)

# 处理 block03
for dataset in block_datasets:
    df = process_dataset_logs(dataset, "block03_log")
    df.to_csv(f"results_csv/imputation/block05/{dataset}.csv", index=False)

# 处理 block05
for dataset in block_datasets:
    df = process_dataset_logs(dataset, "block05_log")
    df.to_csv(f"results_csv/imputation/block05/{dataset}.csv", index=False)

# subseq05实验
for dataset in ["BeijingAir", "Electricity", "ETT_h1", "ItalyAir", "PeMS"]:
    df = process_dataset_logs(dataset, "subseq05_log")
    os.makedirs("results_csv/imputation/subseq05", exist_ok=True)
    df.to_csv(f"results_csv/imputation/subseq05/{dataset}.csv", index=False)

print("所有结果已处理完成！")