"""
综合排名脚本：基于MAE、MSE、MRE三个指标计算模型排名
忽略未完成的实验（指标为0）
"""

import os
import pandas as pd
import numpy as np
from collections import defaultdict

BASE_PATH = "results_csv/imputation"

# 定义所有实验配置
EXPERIMENTS = {
    'point01': ["BeijingAir", "Electricity", "ETT_h1", "ItalyAir", "PeMS", "PhysioNet2012"],
    'point05': ["BeijingAir", "Electricity", "ETT_h1", "ItalyAir", "PeMS"],
    'point09': ["BeijingAir", "Electricity", "ETT_h1", "ItalyAir", "PeMS"],
    'block05': ["BeijingAir", "Electricity", "ETT_h1", "ItalyAir", "PeMS"],
    'subseq05': ["BeijingAir", "Electricity", "ETT_h1", "ItalyAir", "PeMS"],
}

def parse_metric_value(value_str):
    """
    解析指标字符串，格式: "0.123 (0.004)"
    返回平均值，如果是"0"则返回None表示未完成
    """
    if pd.isna(value_str) or value_str == "0":
        return None
    
    try:
        # 提取括号前的数值
        if '(' in value_str:
            mean_str = value_str.split('(')[0].strip()
            return float(mean_str)
        else:
            return float(value_str)
    except:
        return None

def calculate_rankings(df, metrics=['MAE', 'MSE', 'MRE']):
    """
    计算每个指标的排名
    返回包含排名的DataFrame
    """
    df_copy = df.copy()
    
    # 解析指标值
    for metric in metrics:
        if metric in df_copy.columns:
            df_copy[f'{metric}_value'] = df_copy[metric].apply(parse_metric_value)
    
    # 过滤掉所有指标都为None的行
    valid_mask = df_copy[[f'{m}_value' for m in metrics if f'{m}_value' in df_copy.columns]].notna().any(axis=1)
    df_valid = df_copy[valid_mask].copy()
    
    if len(df_valid) == 0:
        return None
    
    # 计算排名（越小越好，所以ascending=True）
    for metric in metrics:
        value_col = f'{metric}_value'
        rank_col = f'{metric}_Rank'
        if value_col in df_valid.columns:
            df_valid[rank_col] = df_valid[value_col].rank(ascending=True, method='min')
    
    # 计算平均排名作为综合排名
    rank_cols = [f'{m}_Rank' for m in metrics if f'{m}_Rank' in df_valid.columns]
    if rank_cols:
        df_valid['Avg_Rank'] = df_valid[rank_cols].mean(axis=1)
        df_valid['Overall_Rank'] = df_valid['Avg_Rank'].rank(ascending=True, method='min')
    
    return df_valid

def create_ranking_summary(experiment, datasets):
    """
    为某个实验配置创建排名汇总
    """
    all_rankings = []
    
    for dataset in datasets:
        csv_path = os.path.join(BASE_PATH, experiment, f"{dataset}.csv")
        
        if not os.path.exists(csv_path):
            print(f"  跳过: {csv_path} 不存在")
            continue
        
        # 读取CSV
        df = pd.read_csv(csv_path)
        
        # 计算排名
        df_ranked = calculate_rankings(df)
        
        if df_ranked is None or len(df_ranked) == 0:
            print(f"  跳过: {dataset} 没有有效数据")
            continue
        
        # 添加数据集信息
        df_ranked['Dataset'] = dataset
        
        # 选择需要的列
        cols = ['Dataset', 'Model', 'Size', 'MAE', 'MSE', 'MRE', 
                'MAE_Rank', 'MSE_Rank', 'MRE_Rank', 'Avg_Rank', 'Overall_Rank']
        cols = [c for c in cols if c in df_ranked.columns]
        
        all_rankings.append(df_ranked[cols])
    
    if not all_rankings:
        return None
    
    # 合并所有数据集
    combined = pd.concat(all_rankings, ignore_index=True)
    
    # 按综合排名排序
    if 'Overall_Rank' in combined.columns:
        combined = combined.sort_values(['Dataset', 'Overall_Rank'])
    
    return combined

def create_overall_ranking_across_datasets(experiment, datasets):
    """
    创建跨数据集的综合排名
    """
    model_scores = defaultdict(lambda: {'ranks': [], 'valid_count': 0})
    
    for dataset in datasets:
        csv_path = os.path.join(BASE_PATH, experiment, f"{dataset}.csv")
        
        if not os.path.exists(csv_path):
            continue
        
        df = pd.read_csv(csv_path)
        df_ranked = calculate_rankings(df)
        
        if df_ranked is None or len(df_ranked) == 0:
            continue
        
        # 收集每个模型的排名
        for _, row in df_ranked.iterrows():
            model = row['Model']
            if 'Avg_Rank' in row:
                model_scores[model]['ranks'].append(row['Avg_Rank'])
                model_scores[model]['valid_count'] += 1
    
    # 计算平均排名
    ranking_data = []
    for model, data in model_scores.items():
        if data['valid_count'] > 0:
            avg_rank = np.mean(data['ranks'])
            ranking_data.append({
                'Model': model,
                'Avg_Rank_Across_Datasets': avg_rank,
                'Valid_Datasets': data['valid_count'],
                'Total_Datasets': len(datasets)
            })
    
    if not ranking_data:
        return None
    
    df_overall = pd.DataFrame(ranking_data)
    df_overall['Overall_Rank'] = df_overall['Avg_Rank_Across_Datasets'].rank(ascending=True, method='min')
    df_overall = df_overall.sort_values('Overall_Rank')
    
    return df_overall

def main():
    print("=" * 80)
    print("开始生成综合排名")
    print("=" * 80)
    
    for experiment, datasets in EXPERIMENTS.items():
        print(f"\n处理实验配置: {experiment}")
        print("-" * 80)
        
        # 1. 创建每个数据集的详细排名
        df_detailed = create_ranking_summary(experiment, datasets)
        
        if df_detailed is not None:
            output_dir = os.path.join(BASE_PATH, experiment)
            os.makedirs(output_dir, exist_ok=True)
            
            output_file = os.path.join(output_dir, "rankings_detailed.csv")
            df_detailed.to_csv(output_file, index=False)
            print(f"  ✓ 详细排名保存到: {output_file}")
            
            # 打印前几行预览
            print(f"\n  预览 (前5行):")
            print(df_detailed.head().to_string(index=False))
        else:
            print(f"  ✗ 没有有效数据")
        
        # 2. 创建跨数据集的综合排名
        df_overall = create_overall_ranking_across_datasets(experiment, datasets)
        
        if df_overall is not None:
            output_file = os.path.join(BASE_PATH, experiment, "rankings_overall.csv")
            df_overall.to_csv(output_file, index=False)
            print(f"\n  ✓ 综合排名保存到: {output_file}")
            
            print(f"\n  综合排名:")
            print(df_overall.to_string(index=False))
    
    # 3. 创建所有实验配置的总排名
    print("\n" + "=" * 80)
    print("创建全局综合排名（跨所有实验配置）")
    print("=" * 80)
    
    global_model_scores = defaultdict(lambda: {'ranks': [], 'valid_count': 0, 'total_count': 0})
    
    for experiment, datasets in EXPERIMENTS.items():
        for dataset in datasets:
            csv_path = os.path.join(BASE_PATH, experiment, f"{dataset}.csv")
            
            if not os.path.exists(csv_path):
                continue
            
            df = pd.read_csv(csv_path)
            df_ranked = calculate_rankings(df)
            
            if df_ranked is None or len(df_ranked) == 0:
                continue
            
            for _, row in df_ranked.iterrows():
                model = row['Model']
                global_model_scores[model]['total_count'] += 1
                if 'Avg_Rank' in row:
                    global_model_scores[model]['ranks'].append(row['Avg_Rank'])
                    global_model_scores[model]['valid_count'] += 1
    
    # 计算全局排名
    global_ranking_data = []
    for model, data in global_model_scores.items():
        if data['valid_count'] > 0:
            avg_rank = np.mean(data['ranks'])
            global_ranking_data.append({
                'Model': model,
                'Avg_Rank': avg_rank,
                'Valid_Experiments': data['valid_count'],
                'Total_Experiments': data['total_count']
            })
    
    if global_ranking_data:
        df_global = pd.DataFrame(global_ranking_data)
        df_global['Global_Rank'] = df_global['Avg_Rank'].rank(ascending=True, method='min')
        df_global = df_global.sort_values('Global_Rank')
        
        output_file = os.path.join(BASE_PATH, "rankings_global.csv")
        df_global.to_csv(output_file, index=False)
        print(f"\n✓ 全局排名保存到: {output_file}")
        
        print(f"\n全局综合排名:")
        print(df_global.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("排名生成完成！")
    print("=" * 80)
    print(f"\n输出文件说明:")
    print(f"  - results_csv/imputation/<experiment>/rankings_detailed.csv: 每个数据集的详细排名")
    print(f"  - results_csv/imputation/<experiment>/rankings_overall.csv: 该实验配置下的综合排名")
    print(f"  - results_csv/imputation/rankings_global.csv: 全局综合排名（跨所有配置）")

if __name__ == "__main__":
    main()