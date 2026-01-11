"""
高级分析脚本：
1. 按缺失模式分析
2. 消融实验按数据集/模式分解
3. vs Naive 改进百分比
"""

import os
import pandas as pd
import numpy as np
from collections import defaultdict

BASE_PATH = "results_csv/imputation"
NAIVE_CSV_PATH = "results_csv/naive_imputation.csv"

# 配置（复用上面的映射）
DATASET_NAME_MAP = {
    'beijing_air_quality': 'BeijingAir',
    'electricity_load_diagrams': 'Electricity',
    'ett': 'ETT_h1',
    'italy_air_quality': 'ItalyAir',
    'pems_traffic': 'PeMS',
    'physionet_2012': 'PhysioNet2012',
}

PATTERN_MAP = {
    'point_rate01': 'point01',
    'point_rate05': 'point05',
    'point_rate09': 'point09',
    'block_rate05': 'block05',
    'subseq_rate05': 'subseq05',
}

NAIVE_METHOD_MAP = {
    'mean': 'Naive_Mean',
    'median': 'Naive_Median',
    'LOCF': 'Naive_LOCF',
    'linear interpolation': 'Naive_LinearInterp',
}

EXPERIMENTS = {
    'point01': ["BeijingAir", "Electricity", "ETT_h1", "ItalyAir", "PeMS", "PhysioNet2012"],
    'point05': ["BeijingAir", "Electricity", "ETT_h1", "ItalyAir", "PeMS"],
    'point09': ["BeijingAir", "Electricity", "ETT_h1", "ItalyAir", "PeMS"],
    'block05': ["BeijingAir", "Electricity", "ETT_h1", "ItalyAir", "PeMS"],
    'subseq05': ["BeijingAir", "Electricity", "ETT_h1", "ItalyAir", "PeMS"],
}

ABLATION_MODELS = ['HELIX', 'HELIX_NoFeatureEmbed', 'HELIX_NoFusion', 'HELIX_NoHybrid', 'HELIX_NoRotaryPE']
KEY_BASELINES = ['ImputeFormer', 'SAITS', 'NonstationaryTransformer', 'PatchTST']

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

def load_naive_results():
    """加载 naive baseline 结果"""
    df = pd.read_csv(NAIVE_CSV_PATH)
    naive_results = defaultdict(dict)
    
    for _, row in df.iterrows():
        pattern_raw = row['type']
        dataset_raw = row['dataset']
        
        pattern = PATTERN_MAP.get(pattern_raw)
        if pattern is None:
            continue
        
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

def load_all_results(naive_results):
    """
    加载所有结果（DL + Naive）
    返回: {(pattern, dataset): DataFrame with Model, MAE, MSE, MRE}
    """
    all_results = {}
    
    for experiment, datasets in EXPERIMENTS.items():
        for dataset in datasets:
            csv_path = os.path.join(BASE_PATH, experiment, f"{dataset}.csv")
            if not os.path.exists(csv_path):
                continue
            
            df = pd.read_csv(csv_path)
            
            # 解析 DL 结果
            data = []
            for _, row in df.iterrows():
                mae = parse_metric_value(row['MAE'])
                mse = parse_metric_value(row['MSE'])
                mre = parse_metric_value(row['MRE'])
                if mae is not None:
                    data.append({
                        'Model': row['Model'],
                        'MAE': mae,
                        'MSE': mse,
                        'MRE': mre,
                    })
            
            # 添加 Naive 结果
            naive_data = naive_results.get((experiment, dataset), {})
            for method, metrics in naive_data.items():
                data.append({
                    'Model': method,
                    'MAE': metrics['MAE'],
                    'MSE': metrics['MSE'],
                    'MRE': metrics['MRE'],
                })
            
            if data:
                all_results[(experiment, dataset)] = pd.DataFrame(data)
    
    return all_results

# ============== 分析 1: 按缺失模式分析 ==============

def analysis_by_pattern(all_results):
    """
    按缺失模式分析各模型的平均排名
    输出: 每种模式下各模型的平均排名
    """
    pattern_ranks = defaultdict(lambda: defaultdict(list))
    
    for (experiment, dataset), df in all_results.items():
        # 计算排名
        df = df.copy()
        for metric in ['MAE', 'MSE', 'MRE']:
            df[f'{metric}_Rank'] = df[metric].rank(ascending=True, method='min')
        df['Avg_Rank'] = df[['MAE_Rank', 'MSE_Rank', 'MRE_Rank']].mean(axis=1)
        
        for _, row in df.iterrows():
            pattern_ranks[experiment][row['Model']].append(row['Avg_Rank'])
    
    # 汇总为表格
    results = []
    all_models = set()
    for pattern_data in pattern_ranks.values():
        all_models.update(pattern_data.keys())
    
    for model in sorted(all_models):
        row = {'Model': model}
        for pattern in ['point01', 'point05', 'point09', 'block05', 'subseq05']:
            ranks = pattern_ranks[pattern].get(model, [])
            if ranks:
                row[pattern] = round(np.mean(ranks), 2)
            else:
                row[pattern] = None
        
        # 计算跨模式的平均和标准差
        valid_ranks = [row[p] for p in ['point01', 'point05', 'point09', 'block05', 'subseq05'] if row.get(p) is not None]
        if valid_ranks:
            row['Avg_Across_Patterns'] = round(np.mean(valid_ranks), 2)
            row['Std_Across_Patterns'] = round(np.std(valid_ranks), 2)
        
        results.append(row)
    
    df_result = pd.DataFrame(results)
    df_result = df_result.sort_values('Avg_Across_Patterns')
    
    return df_result

# ============== 分析 2: 消融实验按数据集分解 ==============

def analysis_ablation_by_dataset(all_results):
    """
    消融实验按数据集分解
    展示每个消融变体在各数据集上相对 HELIX 的性能变化
    """
    results = []
    
    datasets_all = set()
    for (_, dataset) in all_results.keys():
        datasets_all.add(dataset)
    
    for ablation in ['HELIX_NoFeatureEmbed', 'HELIX_NoFusion', 'HELIX_NoHybrid', 'HELIX_NoRotaryPE']:
        row = {'Ablation': ablation.replace('HELIX_', 'w/o ')}
        
        for dataset in sorted(datasets_all):
            # 收集该数据集在所有模式下的平均排名差
            rank_diffs = []
            
            for (experiment, ds), df in all_results.items():
                if ds != dataset:
                    continue
                
                df = df.copy()
                df['MAE_Rank'] = df['MAE'].rank(ascending=True, method='min')
                
                helix_rank = df[df['Model'] == 'HELIX']['MAE_Rank'].values
                ablation_rank = df[df['Model'] == ablation]['MAE_Rank'].values
                
                if len(helix_rank) > 0 and len(ablation_rank) > 0:
                    rank_diffs.append(ablation_rank[0] - helix_rank[0])
            
            if rank_diffs:
                avg_diff = np.mean(rank_diffs)
                row[dataset] = f"+{avg_diff:.1f}" if avg_diff > 0 else f"{avg_diff:.1f}"
            else:
                row[dataset] = "N/A"
        
        results.append(row)
    
    return pd.DataFrame(results)

# ============== 分析 3: 消融实验按缺失模式分解 ==============

def analysis_ablation_by_pattern(all_results):
    """
    消融实验按缺失模式分解
    """
    results = []
    
    for ablation in ['HELIX_NoFeatureEmbed', 'HELIX_NoFusion', 'HELIX_NoHybrid', 'HELIX_NoRotaryPE']:
        row = {'Ablation': ablation.replace('HELIX_', 'w/o ')}
        
        for pattern in ['point01', 'point05', 'point09', 'block05', 'subseq05']:
            rank_diffs = []
            
            for (experiment, dataset), df in all_results.items():
                if experiment != pattern:
                    continue
                
                df = df.copy()
                df['MAE_Rank'] = df['MAE'].rank(ascending=True, method='min')
                
                helix_rank = df[df['Model'] == 'HELIX']['MAE_Rank'].values
                ablation_rank = df[df['Model'] == ablation]['MAE_Rank'].values
                
                if len(helix_rank) > 0 and len(ablation_rank) > 0:
                    rank_diffs.append(ablation_rank[0] - helix_rank[0])
            
            if rank_diffs:
                avg_diff = np.mean(rank_diffs)
                row[pattern] = f"+{avg_diff:.1f}" if avg_diff > 0 else f"{avg_diff:.1f}"
            else:
                row[pattern] = "N/A"
        
        results.append(row)
    
    return pd.DataFrame(results)

# ============== 分析 4: vs Naive 改进百分比 ==============

def analysis_vs_naive(all_results):
    """
    计算 HELIX 和关键 baseline 相对 Naive (Linear Interpolation) 的改进百分比
    """
    results = []
    models_to_compare = ['HELIX',
    'HELIX_NoFeatureEmbed',
    'HELIX_NoFusion',
    'HELIX_NoHybrid',
    'HELIX_NoRotaryPE',
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
    'PatchTST',]
    
    for (experiment, dataset), df in all_results.items():
        # 获取 Linear Interpolation 的 MAE
        naive_mae = df[df['Model'] == 'Naive_LinearInterp']['MAE'].values
        if len(naive_mae) == 0:
            continue
        naive_mae = naive_mae[0]
        
        row = {
            'Pattern': experiment,
            'Dataset': dataset,
            'Naive_LinearInterp_MAE': round(naive_mae, 3),
        }
        
        for model in models_to_compare:
            model_mae = df[df['Model'] == model]['MAE'].values
            if len(model_mae) > 0:
                improvement = (naive_mae - model_mae[0]) / naive_mae * 100
                row[f'{model}_MAE'] = round(model_mae[0], 3)
                row[f'{model}_Improv%'] = round(improvement, 1)
            else:
                row[f'{model}_MAE'] = None
                row[f'{model}_Improv%'] = None
        
        results.append(row)
    
    df_result = pd.DataFrame(results)
    
    # 汇总：各模型的平均改进百分比
    summary = {'Model': [], 'Avg_Improvement_vs_LinearInterp': []}
    for model in models_to_compare:
        col = f'{model}_Improv%'
        if col in df_result.columns:
            avg_improv = df_result[col].dropna().mean()
            summary['Model'].append(model)
            summary['Avg_Improvement_vs_LinearInterp'].append(round(avg_improv, 1))
    
    df_summary = pd.DataFrame(summary)
    df_summary = df_summary.sort_values('Avg_Improvement_vs_LinearInterp', ascending=False)
    
    # 汇总（排除 Electricity）
    summary_no_elec = {'Model': [], 'Avg_Improvement_vs_LinearInterp (excl. Electricity)': []}
    df_no_elec = df_result[df_result['Dataset'] != 'Electricity']
    for model in models_to_compare:
        col = f'{model}_Improv%'
        if col in df_no_elec.columns:
            avg_improv = df_no_elec[col].dropna().mean()
            summary_no_elec['Model'].append(model)
            summary_no_elec['Avg_Improvement_vs_LinearInterp (excl. Electricity)'].append(round(avg_improv, 1))
    
    df_summary_no_elec = pd.DataFrame(summary_no_elec)
    df_summary_no_elec = df_summary_no_elec.sort_values('Avg_Improvement_vs_LinearInterp (excl. Electricity)', ascending=False)
    
    return df_result, df_summary, df_summary_no_elec

# ============== 分析 5: HELIX 胜率统计（修复版） ==============

def analysis_win_rate_debug(all_results):
    """
    Debug 版本：详细打印所有比较
    """
    baselines = [
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
        'Naive_Mean',
        'Naive_Median', 
        'Naive_LOCF',
        'Naive_LinearInterp',
    ]
    
    win_counts = defaultdict(lambda: {'win': 0, 'tie': 0, 'lose': 0, 'total': 0, 'lose_cases': [], 'tie_cases': []})
    
    for (experiment, dataset), df in all_results.items():
        helix_rows = df[df['Model'] == 'HELIX']
        if len(helix_rows) == 0:
            print(f"WARNING: No HELIX in {experiment}/{dataset}")
            continue
        
        helix_mae = float(helix_rows['MAE'].values[0])  # 确保是 float
        
        for baseline in baselines:
            baseline_rows = df[df['Model'] == baseline]
            if len(baseline_rows) == 0:
                continue
            
            baseline_mae = float(baseline_rows['MAE'].values[0])  # 确保是 float
            
            win_counts[baseline]['total'] += 1
            
            # 使用一个小的 epsilon 来处理浮点数比较
            epsilon = 1e-9
            diff = helix_mae - baseline_mae
            
            if diff < -epsilon:  # HELIX 更小 = win
                win_counts[baseline]['win'] += 1
            elif diff > epsilon:  # HELIX 更大 = lose
                win_counts[baseline]['lose'] += 1
                win_counts[baseline]['lose_cases'].append({
                    'pattern': experiment,
                    'dataset': dataset,
                    'HELIX_MAE': helix_mae,
                    'Baseline_MAE': baseline_mae,
                    'diff': diff,
                })
            else:  # 几乎相等 = tie
                win_counts[baseline]['tie'] += 1
                win_counts[baseline]['tie_cases'].append({
                    'pattern': experiment,
                    'dataset': dataset,
                    'HELIX_MAE': helix_mae,
                    'Baseline_MAE': baseline_mae,
                })
    
    # 打印结果
    results = []
    for baseline in baselines:
        counts = win_counts[baseline]
        if counts['total'] > 0:
            win_rate = counts['win'] / counts['total'] * 100
            results.append({
                'Baseline': baseline,
                'Win': counts['win'],
                'Tie': counts['tie'],
                'Lose': counts['lose'],
                'Total': counts['total'],
                'Win_Rate%': round(win_rate, 1),
            })
    
    df_result = pd.DataFrame(results)
    df_result = df_result.sort_values('Win_Rate%', ascending=False)
    
    # 打印详细的 lose 和 tie cases
    print("\n" + "=" * 80)
    print("TIE CASES (详细)")
    print("=" * 80)
    for baseline, counts in win_counts.items():
        if counts['tie'] > 0:
            print(f"\n--- vs {baseline} (Tie {counts['tie']} times) ---")
            for case in counts['tie_cases']:
                print(f"  {case['pattern']}/{case['dataset']}: HELIX={case['HELIX_MAE']:.6f}, {baseline}={case['Baseline_MAE']:.6f}")
    
    print("\n" + "=" * 80)
    print("LOSE CASES (详细)")
    print("=" * 80)
    for baseline, counts in win_counts.items():
        if counts['lose'] > 0:
            print(f"\n--- vs {baseline} (Lose {counts['lose']} times) ---")
            for case in counts['lose_cases']:
                print(f"  {case['pattern']}/{case['dataset']}: HELIX={case['HELIX_MAE']:.4f} > {baseline}={case['Baseline_MAE']:.4f} (diff={case['diff']:.4f})")
    
    return df_result

def main():
    print("=" * 80)
    print("加载数据")
    print("=" * 80)
    
    naive_results = load_naive_results()
    all_results = load_all_results(naive_results)
    print(f"加载了 {len(all_results)} 个实验配置")
    
    output_dir = os.path.join(BASE_PATH, "analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    # 分析 1: 按缺失模式
    print("\n" + "=" * 80)
    print("分析 1: 按缺失模式分析")
    print("=" * 80)
    df_pattern = analysis_by_pattern(all_results)
    df_pattern.to_csv(os.path.join(output_dir, "analysis_by_pattern.csv"), index=False)
    print(df_pattern.head(15).to_string(index=False))
    
    # 分析 2: 消融按数据集
    print("\n" + "=" * 80)
    print("分析 2: 消融实验按数据集分解")
    print("=" * 80)
    df_ablation_dataset = analysis_ablation_by_dataset(all_results)
    df_ablation_dataset.to_csv(os.path.join(output_dir, "analysis_ablation_by_dataset.csv"), index=False)
    print(df_ablation_dataset.to_string(index=False))
    
    # 分析 3: 消融按模式
    print("\n" + "=" * 80)
    print("分析 3: 消融实验按缺失模式分解")
    print("=" * 80)
    df_ablation_pattern = analysis_ablation_by_pattern(all_results)
    df_ablation_pattern.to_csv(os.path.join(output_dir, "analysis_ablation_by_pattern.csv"), index=False)
    print(df_ablation_pattern.to_string(index=False))
    
    # 分析 4: vs Naive
    print("\n" + "=" * 80)
    print("分析 4: vs Naive 改进百分比")
    print("=" * 80)
    df_vs_naive, df_vs_naive_summary, df_vs_naive_summary_no_elec = analysis_vs_naive(all_results)
    df_vs_naive_summary_no_elec.to_csv(os.path.join(output_dir, "analysis_vs_naive_summary_no_electricity.csv"), index=False)
    print("\n汇总（排除 Electricity）:")
    print(df_vs_naive_summary_no_elec.to_string(index=False))
    df_vs_naive.to_csv(os.path.join(output_dir, "analysis_vs_naive_detailed.csv"), index=False)
    df_vs_naive_summary.to_csv(os.path.join(output_dir, "analysis_vs_naive_summary.csv"), index=False)
    print("汇总:")
    print(df_vs_naive_summary.to_string(index=False))
    
    # 分析 5: 胜率
    print("\n" + "=" * 80)
    print("分析 5: HELIX vs Baselines 胜率")
    print("=" * 80)
    df_win_rate = analysis_win_rate_debug(all_results)
    df_win_rate.to_csv(os.path.join(output_dir, "analysis_win_rate.csv"), index=False)
    print(df_win_rate.to_string(index=False))
    
    print("\n" + "=" * 80)
    print(f"所有分析结果已保存到: {output_dir}/")
    print("=" * 80)

if __name__ == "__main__":
    main()