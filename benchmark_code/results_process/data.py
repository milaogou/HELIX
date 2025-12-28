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
    'PatchTST',
    "iTransformer", "SAITS", "NonstationaryTransformer", "ETSformer", "PatchTST", 
    # "Crossformer", "Informer", "Autoformer", "Pyraformer", "Transformer", 
    # "BRITS", "MRNN", "GRUD",
    # "TimesNet", "MICN", "SCINet", 
    # "StemGNN", 
    # "FreTS", "Koopa", "DLinear", "FiLM", 
    # "CSDI", "USGAN", "GPVAE"
]

# 定义模型的config_version映射（用于查找对应的log文件）
MODEL_CONFIG_VERSIONS = {
    # 'HELIX': 'with_LR_decay',  # HELIX的新版本后缀
}

metrics_pattern = re.compile(r"MAE=(\d+\.\d+) ± (\d+\.\d+), MSE=(\d+\.\d+) ± (\d+\.\d+), MRE=(\d+\.\d+) ± (\d+\.\d+), average inference time=(\d+\.\d+)")
params_pattern = re.compile(r"the number of trainable parameters: ([\d,]+)")

def extract_and_format_naive_classification(content):
    imputation_methods = ['Mean', 'Median', 'LOCF', 'Linear']
    data = {
        "methods": ['PR_AUC w XGB', 'PR_AUC w RNN', 'PR_AUC w Transformer', 
                    'ROC_AUC w XGB', 'ROC_AUC w RNN', 'ROC_AUC w Transformer']
    }
    formatted_data = {method: [] for method in imputation_methods}
    
    current_method_index = 0
    for line in content:
        if match:= re.match(r"(\w+)\s+with\s+\w+\s+imputation\s+PR_AUC:\s+([\d.]+)±([\d.]+),\s+ROC_AUC:\s+([\d.]+)±([\d.]+)", line):
            method, pr_auc_mean, pr_auc_std, roc_auc_mean, roc_auc_std = match.groups()
            formatted_data[imputation_methods[current_method_index]].append(f"{float(pr_auc_mean):.3f} ({float(pr_auc_std):.3f})")
            formatted_data[imputation_methods[current_method_index]].append(f"{float(roc_auc_mean):.3f} ({float(roc_auc_std):.3f})")
            
            # Move to the next imputation method after every three lines
            if len(formatted_data[imputation_methods[current_method_index]]) == 6:
                current_method_index += 1
    
    # Convert to DataFrame for better visualization
    final_data = {"name": [], "PR_AUC w XGB": [], "PR_AUC w RNN": [], "PR_AUC w Transformer": [], 
                  "ROC_AUC w XGB": [], "ROC_AUC w RNN": [], "ROC_AUC w Transformer": []}
    
    for method in imputation_methods:
        final_data["name"].append(method)
        final_data["PR_AUC w XGB"].append(formatted_data[method][0])
        final_data["PR_AUC w RNN"].append(formatted_data[method][2])
        final_data["PR_AUC w Transformer"].append(formatted_data[method][4])
        final_data["ROC_AUC w XGB"].append(formatted_data[method][1])
        final_data["ROC_AUC w RNN"].append(formatted_data[method][3])
        final_data["ROC_AUC w Transformer"].append(formatted_data[method][5])
    
    return final_data
def process_dataset_logs(dataset, log_pattern):
    """处理单个数据集的日志文件"""
    log_dir = f"./imputation_log/{log_pattern}/{dataset}_log"
    
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
        
        # 构建文件名，如果有config_version则添加后缀
        if config_version:
            file_name = f"{model}_{dataset}_{config_version}.log"
        else:
            file_name = f"{model}_{dataset}.log"
        
        file_path = os.path.join(log_dir, file_name)
        
        if os.path.exists(file_path):
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
    os.makedirs("./results_csv/imputation/point01", exist_ok=True)
    df.to_csv(f"./results_csv/imputation/point01/{dataset}.csv", index=False)

# point05实验
for dataset in ["BeijingAir", "Electricity", "ETT_h1", "ItalyAir", "Pedestrian", "PeMS"]:
    df = process_dataset_logs(dataset, "point05_log")
    os.makedirs("./results_csv/imputation/point05", exist_ok=True)
    df.to_csv(f"./results_csv/imputation/point05/{dataset}.csv", index=False)

# point09实验
for dataset in ["BeijingAir", "Electricity", "ETT_h1", "ItalyAir", "Pedestrian", "PeMS"]:
    df = process_dataset_logs(dataset, "point09_log")
    os.makedirs("./results_csv/imputation/point09", exist_ok=True)
    df.to_csv(f"./results_csv/imputation/point09/{dataset}.csv", index=False)

# block05实验
for dataset in ["BeijingAir", "Electricity", "ETT_h1", "ItalyAir", "PeMS"]:
    df = process_dataset_logs(dataset, "block05_log")
    os.makedirs("./results_csv/imputation/block05", exist_ok=True)
    df.to_csv(f"./results_csv/imputation/block05/{dataset}.csv", index=False)

# subseq05实验
for dataset in ["BeijingAir", "Electricity", "ETT_h1", "ItalyAir", "Pedestrian", "PeMS"]:
    df = process_dataset_logs(dataset, "subseq05_log")
    os.makedirs("./results_csv/imputation/subseq05", exist_ok=True)
    df.to_csv(f"./results_csv/imputation/subseq05/{dataset}.csv", index=False)

print("所有结果已处理完成！")