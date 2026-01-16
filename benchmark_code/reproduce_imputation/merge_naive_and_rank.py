"""
合并 Naive baseline 和 DL 模型结果，生成含 22 个方法的排名
"""

import os
import pandas as pd
import numpy as np
import re
from collections import defaultdict
# ============== 配置 ==============

# 日志目录映射 (可以是单个目录或多个目录的列表)
LOG_DIR_MAP = {
    'point01': ['point01_log'],
    'point05': ['point05_log'],
    'point09': ['point09_log'],
    'block05': ['block00_log', 'block03_log'],  # 两个目录合并
    'subseq05': ['subseq05_log'],
}

# 模型列表
MODEL_NAMES = [
    "HELIX", "HELIX_NoFeatureEmbed", "HELIX_NoFusion", "HELIX_NoHybrid", "HELIX_NoRotaryPE",
    "TEFN", "TimeMixerPP", "TimeLLM", "MOMENT", "TimeMixer", "ModernTCN",'StemGNN',
    "ImputeFormer", "TOTEM", "iTransformer", "SAITS", "FreTS",
    "NonstationaryTransformer", "PatchTST"
]

# 日志根目录
LOG_ROOT = "."  # 如果日志目录在其他位置，修改这里


BASE_PATH = "results_csv/imputation"
NAIVE_CSV_PATH = "results_csv/naive_imputation.csv"

# 数据集名称映射
DATASET_NAME_MAP = {
    'beijing_air_quality': 'BeijingAir',
    'electricity_load_diagrams': 'Electricity',
    'ett': 'ETT_h1',
    'italy_air_quality': 'ItalyAir',
    'pems_traffic': 'PeMS',
    'physionet_2012': 'PhysioNet2012',
}

# 缺失模式映射
PATTERN_MAP = {
    'point_rate01': 'point01',
    'point_rate05': 'point05',
    'point_rate09': 'point09',
    'block_rate05': 'block05',
    'subseq_rate05': 'subseq05',
}

# Naive 方法名称标准化
NAIVE_METHOD_MAP = {
    'mean': 'Naive_Mean',
    'median': 'Naive_Median',
    'LOCF': 'Naive_LOCF',
    'linear interpolation': 'Naive_LinearInterp',
}

# 实验配置
EXPERIMENTS = {
    'point01': ["BeijingAir",  "ETT_h1", "ItalyAir", "PeMS", "PhysioNet2012"],
    'point05': ["BeijingAir",  "ETT_h1", "ItalyAir", "PeMS"],
    'point09': ["BeijingAir", "ETT_h1", "ItalyAir", "PeMS"],
    'block05': ["BeijingAir",  "ETT_h1", "ItalyAir", "PeMS"],
    'subseq05': ["BeijingAir",  "ETT_h1", "ItalyAir", "PeMS"],
}

def parse_logs_to_csv():
    """
    从日志文件解析结果并生成 CSV
    """
    # 正则表达式 - 匹配最终结果行
    # 示例: Averaged HELIX (803,451 params) on ETT_h1: MAE=0.1278 ± 0.0045..., MSE=..., MRE=..., average inference time=0.06
    metrics_pattern = re.compile(
        r'MAE=([\d.]+)\s*±\s*([\d.eE+-]+),\s*'
        r'MSE=([\d.]+)\s*±\s*([\d.eE+-]+),\s*'
        r'MRE=([\d.]+)\s*±\s*([\d.eE+-]+),\s*'
        r'average inference time=([\d.]+)'
    )
    params_pattern = re.compile(r'\(([\d,]+)\s*params\)')
    
    for experiment, datasets in EXPERIMENTS.items():
        log_dirs = LOG_DIR_MAP.get(experiment, [])
        if not log_dirs:
            print(f"警告: {experiment} 没有对应的日志目录配置")
            continue
        
        for dataset in datasets:
            results = {
                "Model": [],
                "Size": [],
                "MAE": [],
                "MSE": [],
                "MRE": [],
                "Time": []
            }
            
            for model in MODEL_NAMES:
                # 在多个日志目录中查找
                found = False
                for log_dir_name in log_dirs:
                    log_dir = os.path.join(LOG_ROOT, log_dir_name, f"{dataset}_log")
                    file_path = os.path.join(log_dir, f"{model}_{dataset}.log")
                    
                    if os.path.exists(file_path):
                        with open(file_path, 'r') as file:
                            content = file.read()
                            
                            # 提取参数量
                            params = "0"
                            params_match = params_pattern.search(content)
                            if params_match:
                                params = params_match.group(1)
                            
                            # 提取指标 - 从文件末尾向前找，确保是最终结果
                            metrics_matches = list(metrics_pattern.finditer(content))
                            if metrics_matches:
                                # 取最后一个匹配（最终结果）
                                match = metrics_matches[-1]
                                mae, mae_std, mse, mse_std, mre, mre_std, time = match.groups()
                                results["Model"].append(model)
                                results["Size"].append(params)
                                results["MAE"].append(f"{float(mae):.3f} ({float(mae_std):.3f})")
                                results["MSE"].append(f"{float(mse):.3f} ({float(mse_std):.3f})")
                                results["MRE"].append(f"{float(mre):.3f} ({float(mre_std):.3f})")
                                results["Time"].append(time)
                                found = True
                                break  # 找到了就不用继续在其他目录找
                            else:
                                print(f"警告: {file_path} 存在但无法匹配指标")
                
                if not found:
                    # 所有目录都没找到有效结果
                    results["Model"].append(model)
                    results["Size"].append("0")
                    results["MAE"].append("0")
                    results["MSE"].append("0")
                    results["MRE"].append("0")
                    results["Time"].append("0")
            
            # 保存 CSV
            output_dir = os.path.join(BASE_PATH, experiment)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{dataset}.csv")
            
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False)
            print(f"✓ 生成: {output_path} ({len([m for m in results['MAE'] if m != '0'])}/{len(MODEL_NAMES)} 个模型有结果)")

def load_naive_results():
    """
    加载 naive baseline 结果
    返回: {(pattern, dataset): {method: {mae, mse, mre}}}
    """
    df = pd.read_csv(NAIVE_CSV_PATH)
    naive_results = defaultdict(dict)
    
    for _, row in df.iterrows():
        # 解析 type 和 dataset
        pattern_raw = row['type']
        dataset_raw = row['dataset']
        
        # 映射到标准名称
        pattern = PATTERN_MAP.get(pattern_raw)
        if pattern is None:
            continue
        
        # 从 dataset 字段提取数据集名称
        # 格式: beijing_air_quality_rate01_step24_point
        for raw_name, std_name in DATASET_NAME_MAP.items():
            if dataset_raw.startswith(raw_name):
                dataset = std_name
                break
        else:
            continue
        
        method = NAIVE_METHOD_MAP.get(row['method'])
        if method is None:
            continue
        
        naive_results[(pattern, dataset)][method] = {
            'MAE': row['mae'],
            'MSE': row['mse'],
            'MRE': row['mre'],
        }
    
    return naive_results

def format_metric_with_std(value, std=None):
    """格式化指标值"""
    if std is None or std == 0:
        return f"{value:.3f} (N/A)"
    return f"{value:.3f} ({std:.3f})"

def merge_naive_into_dataset_csv(naive_results):
    """
    将 Naive 结果合并到各数据集 CSV 中
    """
    for experiment, datasets in EXPERIMENTS.items():
        for dataset in datasets:
            csv_path = os.path.join(BASE_PATH, experiment, f"{dataset}.csv")
            
            if not os.path.exists(csv_path):
                print(f"跳过: {csv_path} 不存在")
                continue
            
            # 读取 DL 模型结果
            df = pd.read_csv(csv_path)
            
            # 获取对应的 Naive 结果
            naive_data = naive_results.get((experiment, dataset), {})
            
            if not naive_data:
                print(f"警告: {experiment}/{dataset} 没有 Naive 数据")
                continue
            
            # 添加 Naive 行
            naive_rows = []
            for method, metrics in naive_data.items():
                naive_rows.append({
                    'Model': method,
                    'Size': 'N/A',
                    'MAE': format_metric_with_std(metrics['MAE']),
                    'MSE': format_metric_with_std(metrics['MSE']),
                    'MRE': format_metric_with_std(metrics['MRE']),
                    'Time': 'N/A',
                })
            
            df_naive = pd.DataFrame(naive_rows)
            df_merged = pd.concat([df, df_naive], ignore_index=True)
            
            # 保存合并后的文件
            output_path = os.path.join(BASE_PATH, experiment, f"{dataset}_with_naive.csv")
            df_merged.to_csv(output_path, index=False)
            print(f"✓ 保存: {output_path}")

def parse_metric_value(value_str):
    """解析指标字符串，返回均值"""
    if pd.isna(value_str) or value_str == "0" or value_str == "N/A":
        return None
    try:
        if '(' in str(value_str):
            mean_str = str(value_str).split('(')[0].strip()
            return float(mean_str)
        else:
            return float(value_str)
    except:
        return None

def calculate_global_ranking_with_naive(naive_results):
    """
    计算包含 Naive 的全局排名（22个方法）
    """
    model_scores = defaultdict(lambda: {'ranks': [], 'valid_count': 0})
    
    for experiment, datasets in EXPERIMENTS.items():
        for dataset in datasets:
            # === 加载 DL 模型结果 ===
            csv_path = os.path.join(BASE_PATH, experiment, f"{dataset}.csv")
            if not os.path.exists(csv_path):
                continue
            
            df_dl = pd.read_csv(csv_path)
            
            # === 加载 Naive 结果 ===
            naive_data = naive_results.get((experiment, dataset), {})
            naive_rows = []
            for method, metrics in naive_data.items():
                naive_rows.append({
                    'Model': method,
                    'MAE': metrics['MAE'],
                    'MSE': metrics['MSE'],
                    'MRE': metrics['MRE'],
                })
            
            # === 合并并计算排名 ===
            # 解析 DL 模型的指标
            dl_data = []
            for _, row in df_dl.iterrows():
                mae = parse_metric_value(row['MAE'])
                mse = parse_metric_value(row['MSE'])
                mre = parse_metric_value(row['MRE'])
                if mae is not None:
                    dl_data.append({
                        'Model': row['Model'],
                        'MAE': mae,
                        'MSE': mse,
                        'MRE': mre,
                    })
            
            # 合并
            all_data = dl_data + naive_rows
            if not all_data:
                continue
            
            df_combined = pd.DataFrame(all_data)
            
            # 计算排名
            for metric in ['MAE', 'MSE', 'MRE']:
                df_combined[f'{metric}_Rank'] = df_combined[metric].rank(ascending=True, method='min')
            
            df_combined['Avg_Rank'] = df_combined[['MAE_Rank', 'MSE_Rank', 'MRE_Rank']].mean(axis=1)
            
            # 收集每个模型的排名
            for _, row in df_combined.iterrows():
                model = row['Model']
                model_scores[model]['ranks'].append(row['Avg_Rank'])
                model_scores[model]['valid_count'] += 1
    
    # 计算全局排名
    ranking_data = []
    total_experiments = sum(len(datasets) for datasets in EXPERIMENTS.values())
    
    for model, data in model_scores.items():
        if data['valid_count'] > 0:
            avg_rank = np.mean(data['ranks'])
            ranking_data.append({
                'Model': model,
                'Avg_Rank': round(avg_rank, 2),
                'Valid_Experiments': data['valid_count'],
                'Total_Experiments': total_experiments,
                'Category': get_model_category(model),
            })
    
    df_global = pd.DataFrame(ranking_data)
    df_global['Global_Rank'] = df_global['Avg_Rank'].rank(ascending=True, method='min').astype(int)
    df_global = df_global.sort_values('Global_Rank')
    
    return df_global

def get_model_category(model):
    """获取模型类别"""
    if model.startswith('HELIX') and model != 'HELIX':
        return 'Ablation'
    elif model == 'HELIX':
        return 'Ours'
    elif model.startswith('Naive_'):
        return 'Naive'
    elif model in ['ImputeFormer', 'SAITS']:
        return 'Imputation-specific'
    elif model in ['MOMENT', 'TimeLLM']:
        return 'Foundation'
    elif model in ['TEFN', 'TimeMixer', 'TimeMixerPP', 'ModernTCN', 'TOTEM']:
        return 'Recent (2024)'
    elif model == 'FreTS':
        return 'Frequency-domain'
    else:
        return 'Transformer'

def main():
    print("=" * 80)
    print("Step 0: 从日志解析生成 CSV")
    print("=" * 80)
    parse_logs_to_csv()
    
    print("\n" + "=" * 80)
    print("Step 1: 加载 Naive baseline 结果")
    print("=" * 80)
    naive_results = load_naive_results()
    print(f"加载了 {len(naive_results)} 个 (pattern, dataset) 组合的 Naive 结果")
    
    print("\n" + "=" * 80)
    print("Step 2: 合并 Naive 到各数据集 CSV")
    print("=" * 80)
    merge_naive_into_dataset_csv(naive_results)
    
    print("\n" + "=" * 80)
    print("Step 3: 计算全局排名（含 Naive，22 个方法）")
    print("=" * 80)
    df_global = calculate_global_ranking_with_naive(naive_results)
    
    output_path = os.path.join(BASE_PATH, "rankings_global_with_naive.csv")
    df_global.to_csv(output_path, index=False)
    print(f"\n✓ 全局排名保存到: {output_path}")
    print("\n全局排名（Top 22）:")
    print(df_global.to_string(index=False))

if __name__ == "__main__":
    main()