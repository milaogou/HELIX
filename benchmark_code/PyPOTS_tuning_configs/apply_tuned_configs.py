import os
import json
from pathlib import Path

TUNING_OUTPUT_PATH = "hyperparameter_tuning_results"
HPO_RESULTS_PATH = "/home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/hpo_results"

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
    'HELIX_NoFeatureEmbed': 'HELIX_NoFeatureEmbed',
    'HELIX_NoFusion': 'HELIX_NoFusion',
    'HELIX_NoHybrid': 'HELIX_NoHybrid',
    'HELIX_NoRotaryPE': 'HELIX_NoRotaryPE',
    'TEFN': 'TEFN',
    'TimeMixerPP': 'TimeMixerPP',
    'TimeMixer': 'TimeMixer',
    'ModernTCN': 'ModernTCN',
    'ImputeFormer': 'ImputeFormer',
    'MOMENT': 'MOMENT',
    'TOTEM': 'TOTEM',
    'TimeLLM': 'TimeLLM',
}

def parse_directory_name(dir_name: str):
    """解析目录名，提取模型名和数据集名"""
    # 移除 _tuning 后缀
    name = dir_name.replace('_tuning', '')
    
    # 先尝试匹配已知的模型名（按长度从长到短）
    sorted_models = sorted(MODEL_MAPPING.keys(), key=len, reverse=True)
    
    for model in sorted_models:
        if name.startswith(model + '_'):
            dataset_name = name[len(model) + 1:]  # +1 是为了跳过下划线
            return model, dataset_name
    
    # 如果没有匹配，尝试用第一个下划线分割（兼容旧逻辑）
    parts = name.split('_', 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    
    return None, None

def load_best_config(tuning_dir: str):
    best_config_path = os.path.join(tuning_dir, "best_config.json")
    if os.path.exists(best_config_path):
        with open(best_config_path, 'r') as f:
            return json.load(f)
    return None

def format_value(value):
    if isinstance(value, str):
        return f"'{value}'"
    elif isinstance(value, bool):
        return str(value)
    elif isinstance(value, list):
        return str(value)
    else:
        return str(value)

def check_model_exists_in_file(file_path: str, model_name: str):
    """检查模型是否已在配置文件中"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return f"'{model_name}':" in content

def add_new_model_to_file(file_path: str, model_name: str, params: dict):
    """在文件末尾添加新模型配置"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 找到最后一个 } 的位置
    last_brace_idx = -1
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() == '}':
            last_brace_idx = i
            break
    
    if last_brace_idx == -1:
        print(f"  ⚠️  无法找到配置文件的结束位置")
        return False
    
    # 在最后一个 } 之前插入新配置
    new_config_lines = [f"    '{model_name}': {{\n"]
    for key, value in params.items():
        new_config_lines.append(f"        '{key}': {format_value(value)},\n")
    new_config_lines.append("    },\n")
    
    lines = lines[:last_brace_idx] + new_config_lines + lines[last_brace_idx:]
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    return True

def update_hpo_file(dataset_file: str, model_name: str, new_params: dict):
    file_path = os.path.join(HPO_RESULTS_PATH, dataset_file)
    
    # 检查模型是否存在
    model_exists = check_model_exists_in_file(file_path, model_name)
    
    if not model_exists:
        print(f"  ℹ️  模型 {model_name} 不存在，将添加新配置")
        return add_new_model_to_file(file_path, model_name, new_params)
    
    # 如果存在，则更新
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        
        if f"'{model_name}':" in line:
            indent = len(line) - len(line.lstrip())
            new_lines.append(line)
            i += 1
            
            # 跳过原有配置
            while i < len(lines):
                if lines[i].strip().startswith('}'):
                    break
                i += 1
            
            # 写入新配置
            for key, value in new_params.items():
                new_lines.append(f"{' ' * (indent + 4)}'{key}': {format_value(value)},\n")
            
            new_lines.append(f"{' ' * indent}}},\n")
            i += 1
        else:
            new_lines.append(line)
            i += 1
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    return True

def main():
    base_path = Path(TUNING_OUTPUT_PATH)
    tuning_dirs = [d for d in base_path.iterdir() if d.is_dir() and '_tuning' in d.name]
    
    print(f"找到 {len(tuning_dirs)} 个调优结果目录\n")
    
    updated_count = 0
    added_count = 0
    failed_count = 0
    
    for tuning_dir in sorted(tuning_dirs):
        dir_name = tuning_dir.name
        
        model_name, dataset_name = parse_directory_name(dir_name)
        
        if model_name is None or dataset_name is None:
            print(f"⚠️  无法解析目录名: {dir_name}")
            failed_count += 1
            continue
        
        if model_name not in MODEL_MAPPING:
            print(f"⚠️  未知模型: {model_name} (来自目录: {dir_name})")
            failed_count += 1
            continue
        
        if dataset_name not in DATASET_MAPPING:
            print(f"⚠️  未知数据集: {dataset_name} (来自目录: {dir_name})")
            failed_count += 1
            continue
        
        print(f"处理 {model_name} on {dataset_name}")
        
        best_config = load_best_config(str(tuning_dir))
        if best_config is None:
            print(f"  ⚠️  未找到best_config.json")
            failed_count += 1
            continue
        
        print(f"  ✓ 加载了 {len(best_config)} 个参数")
        
        dataset_file = DATASET_MAPPING[dataset_name]
        file_path = os.path.join(HPO_RESULTS_PATH, dataset_file)
        model_exists = check_model_exists_in_file(file_path, model_name)
        
        success = update_hpo_file(dataset_file, model_name, best_config)
        
        if success:
            if model_exists:
                print(f"  ✓ 已更新 {dataset_file} 中的 {model_name} 配置\n")
                updated_count += 1
            else:
                print(f"  ✓ 已添加 {dataset_file} 中的 {model_name} 配置\n")
                added_count += 1
        else:
            print(f"  ✗ 更新失败\n")
            failed_count += 1
    
    print("=" * 50)
    print(f"总结:")
    print(f"  更新: {updated_count}")
    print(f"  添加: {added_count}")
    print(f"  失败: {failed_count}")
    print(f"  总计: {len(tuning_dirs)}")

if __name__ == "__main__":
    main()