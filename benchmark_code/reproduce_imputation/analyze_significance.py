"""
25次重复实验的统计显著性检验
"""

import os
import re
import numpy as np
from scipy import stats
import pandas as pd

# ============== 配置 ==============
LOG_DIR = "point05_log/ETT_h1_log"  # 25次重复实验的log目录
OUTPUT_DIR = "results_csv/imputation/analysis"

# 要比较的模型对
COMPARISON_PAIRS = [
    # HELIX vs Imputation-specific
    ('HELIX', 'ImputeFormer'),
    ('HELIX', 'SAITS'),
    # HELIX vs Foundation Models
    ('HELIX', 'MOMENT'),
    ('HELIX', 'TimeLLM'),
    # HELIX vs Recent (2024)
    ('HELIX', 'TEFN'),
    ('HELIX', 'TimeMixer'),
    ('HELIX', 'TimeMixerPP'),
    ('HELIX', 'ModernTCN'),
    ('HELIX', 'StemGNN'),
    ('HELIX', 'TOTEM'),
    # HELIX vs Transformer
    ('HELIX', 'iTransformer'),
    ('HELIX', 'NonstationaryTransformer'),
    ('HELIX', 'PatchTST'),
    # HELIX vs Frequency-domain
    ('HELIX', 'FreTS'),
    # HELIX vs Ablations
    ('HELIX', 'HELIX_NoFeatureEmbed'),
    ('HELIX', 'HELIX_NoFusion'),
    ('HELIX', 'HELIX_NoHybrid'),
    ('HELIX', 'HELIX_NoSinusoidalPE'),
]

def parse_round_results(log_path):
    """
    解析log文件，提取每轮的结果
    返回: {'MAE': [round1, round2, ...], 'MSE': [...], 'MRE': [...]}
    """
    if not os.path.exists(log_path):
        return None
    
    results = {'MAE': [], 'MSE': [], 'MRE': []}
    
    # 匹配格式: Round23 - TEFN on ETT_h1: MAE=0.4703, MSE=0.5006, MRE=0.5554
    round_pattern = re.compile(
        r"Round(\d+) - \w+ on \w+: MAE=([\d.]+), MSE=([\d.]+), MRE=([\d.]+)"
    )
    
    with open(log_path, 'r') as f:
        for line in f:
            match = round_pattern.search(line)
            if match:
                round_num, mae, mse, mre = match.groups()
                results['MAE'].append(float(mae))
                results['MSE'].append(float(mse))
                results['MRE'].append(float(mre))
    
    return results if results['MAE'] else None

def find_log_file(model, dataset="ETT_h1"):
    """查找log文件"""
    possible_names = [
        f"{model}_{dataset}_.log",
        f"{model}_{dataset}.log",
    ]
    
    for name in possible_names:
        path = os.path.join(LOG_DIR, name)
        if os.path.exists(path):
            return path
    
    return None

def perform_significance_test(results1, results2, test_type='wilcoxon'):
    """
    进行统计显著性检验
    test_type: 'ttest' 或 'wilcoxon'
    """
    if test_type == 'ttest':
        # 配对t检验
        statistic, p_value = stats.ttest_rel(results1, results2)
    else:
        # Wilcoxon符号秩检验（非参数）
        statistic, p_value = stats.wilcoxon(results1, results2)
    
    return statistic, p_value

def main():
    print("=" * 80)
    print("25次重复实验统计显著性检验")
    print("=" * 80)
    
    # 加载各模型的结果
    model_results = {}
    
    models_to_load = set()
    for m1, m2 in COMPARISON_PAIRS:
        models_to_load.add(m1)
        models_to_load.add(m2)
    
    for model in models_to_load:
        log_path = find_log_file(model)
        if log_path:
            results = parse_round_results(log_path)
            if results and len(results['MAE']) >= 5:  # 至少5次才有意义
                model_results[model] = results
                print(f"✓ {model}: 找到 {len(results['MAE'])} 轮结果")
            else:
                print(f"✗ {model}: 结果不足")
        else:
            print(f"✗ {model}: 未找到log文件")
    
    if len(model_results) < 2:
        print("\n结果不足，无法进行显著性检验")
        return
    
    # 进行显著性检验
    print("\n" + "=" * 80)
    print("显著性检验结果 (Wilcoxon signed-rank test)")
    print("=" * 80)
    
    significance_results = []
    
    for model1, model2 in COMPARISON_PAIRS:
        if model1 not in model_results or model2 not in model_results:
            continue
        
        r1 = model_results[model1]
        r2 = model_results[model2]
        
        # 确保长度一致
        min_len = min(len(r1['MAE']), len(r2['MAE']))
        
        for metric in ['MAE', 'MSE', 'MRE']:
            vals1 = r1[metric][:min_len]
            vals2 = r2[metric][:min_len]
            
            mean1 = np.mean(vals1)
            mean2 = np.mean(vals2)
            std1 = np.std(vals1)
            std2 = np.std(vals2)
            
            # Wilcoxon 检验
            try:
                stat, p_value = perform_significance_test(vals1, vals2, 'wilcoxon')
            except:
                stat, p_value = None, None
            
            significance_results.append({
                'Model1': model1,
                'Model2': model2,
                'Metric': metric,
                'Model1_Mean': round(mean1, 4),
                'Model1_Std': round(std1, 4),
                'Model2_Mean': round(mean2, 4),
                'Model2_Std': round(std2, 4),
                'Diff': round(mean2 - mean1, 4),
                'p_value': round(p_value, 6) if p_value else None,
                'Significant_0.05': 'Yes' if p_value and p_value < 0.05 else 'No',
                'Significant_0.01': 'Yes' if p_value and p_value < 0.01 else 'No',
            })
    
    df_significance = pd.DataFrame(significance_results)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "analysis_significance.csv")
    df_significance.to_csv(output_path, index=False)
    
    print(df_significance.to_string(index=False))
    print(f"\n✓ 结果保存到: {output_path}")

if __name__ == "__main__":
    main()
