import os
import json
import re
from pathlib import Path

TUNING_OUTPUT_PATH = "hyperparameter_tuning_results"
HPO_RESULTS_PATH = "/home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/hpo_results"

# 数据集名称映射
DATASET_MAPPING = {
    'BeijingAir': 'beijing_air.py',
    'ETT_h1': 'ett_h1.py',
    'Electricity': 'electricity.py',
    'ItalyAir': 'italy_air.py',
    'Pedestrian': 'pedestrian.py',
    'PeMS': 'pems.py',
    'PhysioNet2012': 'physionet2012.py',
    'PhysioNet2019': 'physionet2019.py',
}

MODEL_MAPPING = {
    'HELIX': 'HELIX',
    'TEFN': 'TEFN',
    'TimeMixerPP': 'TimeMixerPP',
    'TimeMixer': 'TimeMixer',
    'ModernTCN': 'ModernTCN',
    'ImputeFormer': 'ImputeFormer',
    'MOMENT': 'MOMENT',
}

def load_best_config(tuning_dir: str):
    best_config_path = os.path.join(tuning_dir, "best_config.json")
    if os.path.exists(best_config_path):
        with open(best_config_path, 'r') as f:
            return json.load(f)
    return None

def update_hpo_file(dataset_file: str, model_name: str, new_params: dict):
    file_path = os.path.join(HPO_RESULTS_PATH, dataset_file)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 找到模型配置块
    pattern = rf"('{model_name}':\s*\{{[^}}]*\}})"
    match = re.search(pattern, content, re.DOTALL)
    
    if not match:
        print(f"  ⚠️  未找到 {model_name} 配置块")
        return False
    
    old_block = match.group(1)
    
    # 解析现有参数
    param_pattern = r"'(\w+)':\s*([^,\n]+)"
    existing_params = dict(re.findall(param_pattern, old_block))
    
    # 更新参数
    updated_params = existing_params.copy()
    for key, value in new_params.items():
        if isinstance(value, str):
            updated_params[key] = f"'{value}'"
        elif isinstance(value, bool):
            updated_params[key] = str(value)
        else:
            updated_params[key] = str(value)
    
    # 重建配置块
    new_block = f"'{model_name}': {{\n"
    for key, value in updated_params.items():
        new_block += f"        '{key}': {value},\n"
    new_block += "    }"
    
    # 替换
    new_content = content.replace(old_block, new_block)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    return True

def main():
    base_path = Path(TUNING_OUTPUT_PATH)
    tuning_dirs = [d for d in base_path.iterdir() if d.is_dir() and '_tuning' in d.name]
    
    print(f"找到 {len(tuning_dirs)} 个调优结果目录\n")
    
    for tuning_dir in sorted(tuning_dirs):
        dir_name = tuning_dir.name.replace('_tuning', '')
        
        # 解析模型和数据集名称
        parts = dir_name.split('_', 1)
        if len(parts) != 2:
            print(f"⚠️  无法解析目录名: {dir_name}")
            continue
        
        model_name, dataset_name = parts
        
        if model_name not in MODEL_MAPPING:
            print(f"⚠️  未知模型: {model_name}")
            continue
        
        if dataset_name not in DATASET_MAPPING:
            print(f"⚠️  未知数据集: {dataset_name}")
            continue
        
        print(f"处理 {model_name} on {dataset_name}")
        
        # 加载最佳配置
        best_config = load_best_config(str(tuning_dir))
        if best_config is None:
            print(f"  ⚠️  未找到best_config.json")
            continue
        
        print(f"  ✓ 加载了 {len(best_config)} 个参数")
        
        # 更新HPO文件
        dataset_file = DATASET_MAPPING[dataset_name]
        success = update_hpo_file(dataset_file, model_name, best_config)
        
        if success:
            print(f"  ✓ 已更新 {dataset_file} 中的 {model_name} 配置\n")
        else:
            print(f"  ✗ 更新失败\n")

if __name__ == "__main__":
    main()