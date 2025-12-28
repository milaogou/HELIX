"""
Collect and summarize out-of-sample evaluation results
Generate CSV tables with mean ± std for each model-dataset-pattern combination
"""

import os
import numpy as np
import pandas as pd
from pypots.data.saving import pickle_load
from collections import defaultdict

OUTPUT_BASE_PATH = "out_sample_eval"

MODELS = ['ImputeFormer', 'TEFN', 'SAITS', 'iTransformer', 'HELIX']

DATASET_PATTERNS = {
    'BeijingAir': [
        'beijing_air_quality_rate00_step24_block_blocklen6',
        'beijing_air_quality_rate01_step24_point',
        'beijing_air_quality_rate05_step24_point',
        'beijing_air_quality_rate05_step24_subseq_seqlen18',
        'beijing_air_quality_rate09_step24_point',
    ],
    'Electricity': [
        'electricity_load_diagrams_rate00_step96_block_blocklen8',
        'electricity_load_diagrams_rate01_step96_point',
        'electricity_load_diagrams_rate05_step96_point',
        'electricity_load_diagrams_rate05_step96_subseq_seqlen72',
        'electricity_load_diagrams_rate09_step96_point',
    ],
    'ETT_h1': [
        'ett_rate01_step48_point',
        'ett_rate03_step48_block_blocklen6',
        'ett_rate05_step48_point',
        'ett_rate05_step48_subseq_seqlen36',
        'ett_rate09_step48_point',
    ],
    'ItalyAir': [
        'italy_air_quality_rate00_step12_block_blocklen4',
        'italy_air_quality_rate01_step12_point',
        'italy_air_quality_rate05_step12_point',
        'italy_air_quality_rate05_step12_subseq_seqlen8',
        'italy_air_quality_rate09_step12_point',
    ],
    'PeMS': [
        'pems_traffic_rate00_step24_block_blocklen6',
        'pems_traffic_rate01_step24_point',
        'pems_traffic_rate05_step24_point',
        'pems_traffic_rate05_step24_subseq_seqlen18',
        'pems_traffic_rate09_step24_point',
    ],
}

TRAIN_PATTERNS = {
    'BeijingAir': 'point05',
    'Electricity': 'point05',
    'ETT_h1': 'point05',
    'ItalyAir': 'point05',
    'PeMS': 'point05',
}

N_ROUNDS = 5

def get_pattern_short_name(pattern_full):
    """Extract short pattern name for logging"""
    # Handle short format like "point05", "block00", "subseq05"
    if not '_' in pattern_full or 'rate' not in pattern_full:
        # Already in short format or simple pattern name
        if 'point' in pattern_full:
            # Extract number after 'point'
            num = pattern_full.replace('point', '')
            return f'point{num}'
        elif 'block' in pattern_full:
            num = pattern_full.replace('block', '')
            return f'block{num}'
        elif 'subseq' in pattern_full:
            num = pattern_full.replace('subseq', '')
            return f'subseq{num}'
        else:
            return pattern_full
    
    # Handle full format like "beijing_air_quality_rate05_step24_point"
    if 'block' in pattern_full:
        rate = pattern_full.split('rate')[1].split('_')[0]
        return f'block{rate}'
    elif 'subseq' in pattern_full:
        rate = pattern_full.split('rate')[1].split('_')[0]
        return f'subseq{rate}'
    elif 'point' in pattern_full:
        rate = pattern_full.split('rate')[1].split('_')[0]
        return f'point{rate}'
    return 'unknown'

def collect_results():
    """Collect all results from pickle files"""
    results = defaultdict(lambda: defaultdict(list))
    
    print("收集Out-of-Sample评估结果...")
    print("=" * 80)
    
    missing_count = 0
    found_count = 0
    
    for dataset, patterns in DATASET_PATTERNS.items():
        train_pattern = TRAIN_PATTERNS[dataset]
        print(f"\n数据集: {dataset} (训练模式: {train_pattern})")
        print("-" * 80)
        
        for model in MODELS:
            model_results = {}
            
            for test_pattern in patterns:
                test_short = get_pattern_short_name(test_pattern)
                key = f"{dataset}_{test_short}"
                
                mae_list, mse_list, mre_list = [], [], []
                
                for round_id in range(N_ROUNDS):
                    result_dir = os.path.join(
                        OUTPUT_BASE_PATH,
                        f"{model}_{dataset}_train_{train_pattern}_test_{test_pattern}",
                        f"round_{round_id}"
                    )
                    result_file = os.path.join(result_dir, "results.pkl")
                    
                    if os.path.exists(result_file):
                        try:
                            data = pickle_load(result_file)
                            mae_list.append(data['mae'])
                            mse_list.append(data['mse'])
                            mre_list.append(data['mre'])
                            found_count += 1
                        except Exception as e:
                            print(f"  [ERROR] 加载失败 {result_file}: {e}")
                            missing_count += 1
                    else:
                        missing_count += 1
                
                if mae_list:
                    results[key][model] = {
                        'mae_mean': np.mean(mae_list),
                        'mae_std': np.std(mae_list),
                        'mse_mean': np.mean(mse_list),
                        'mse_std': np.std(mse_list),
                        'mre_mean': np.mean(mre_list),
                        'mre_std': np.std(mre_list),
                        'n_rounds': len(mae_list),
                    }
                    model_results[test_short] = len(mae_list)
            
            # Print summary for this model
            if model_results:
                results_str = ", ".join([f"{k}:{v}" for k, v in model_results.items()])
                print(f"  ✓ {model:15s}: {results_str}")
            else:
                print(f"  ✗ {model:15s}: 无结果")
    
    print(f"\n{'='*80}")
    print(f"结果收集完成!")
    print(f"找到: {found_count} 个结果文件")
    print(f"缺失: {missing_count} 个结果文件")
    print(f"{'='*80}\n")
    
    return results

def create_summary_tables(results):
    """Create summary tables for each metric"""
    
    for metric in ['mae', 'mse', 'mre']:
        print(f"\n{'='*80}")
        print(f"创建 {metric.upper()} 汇总表...")
        print(f"{'='*80}")
        
        rows = []
        
        for dataset, patterns in DATASET_PATTERNS.items():
            for pattern in patterns:
                pattern_short = get_pattern_short_name(pattern)
                key = f"{dataset}_{pattern_short}"
                
                if key not in results:
                    continue
                
                row_data = {'Dataset': dataset, 'Pattern': pattern_short}
                
                # Mark in-distribution test
                train_pattern = TRAIN_PATTERNS[dataset]
                train_short = get_pattern_short_name(
                    [p for p in patterns if get_pattern_short_name(p) == get_pattern_short_name(train_pattern)][0]
                    if any(get_pattern_short_name(p) == get_pattern_short_name(train_pattern) for p in patterns)
                    else patterns[0]
                )
                is_id = (pattern_short == train_short)
                row_data['Type'] = 'ID' if is_id else 'OOD'
                
                for model in MODELS:
                    if model in results[key]:
                        mean_val = results[key][model][f'{metric}_mean']
                        std_val = results[key][model][f'{metric}_std']
                        n_rounds = results[key][model]['n_rounds']
                        
                        # Format: mean ± std (mark if < 5 rounds)
                        if n_rounds == N_ROUNDS:
                            row_data[model] = f"{mean_val:.4f}±{std_val:.4f}"
                        else:
                            row_data[model] = f"{mean_val:.4f}±{std_val:.4f}*"
                    else:
                        row_data[model] = "N/A"
                
                rows.append(row_data)
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Save to CSV
        output_file = os.path.join(OUTPUT_BASE_PATH, f"summary_{metric.upper()}.csv")
        df.to_csv(output_file, index=False)
        print(f"✓ 保存到: {output_file}")
        
        # Print first few rows
        print(f"\n预览 (前10行):")
        print(df.head(10).to_string(index=False))

def create_summary_tables(results):
    """Create summary tables for each metric"""
    
    for metric in ['mae', 'mse', 'mre']:
        print(f"\n{'='*80}")
        print(f"创建 {metric.upper()} 汇总表...")
        print(f"{'='*80}")
        
        rows = []
        
        for dataset, patterns in DATASET_PATTERNS.items():
            train_pattern = TRAIN_PATTERNS[dataset]
            train_short = get_pattern_short_name(train_pattern)
            
            for pattern in patterns:
                pattern_short = get_pattern_short_name(pattern)
                key = f"{dataset}_{pattern_short}"
                
                if key not in results:
                    continue
                
                row_data = {'Dataset': dataset, 'Pattern': pattern_short}
                
                # Determine if in-distribution
                is_id = is_in_distribution(pattern_short, train_short)
                row_data['Type'] = 'ID' if is_id else 'OOD'
                
                for model in MODELS:
                    if model in results[key]:
                        mean_val = results[key][model][f'{metric}_mean']
                        std_val = results[key][model][f'{metric}_std']
                        n_rounds = results[key][model]['n_rounds']
                        
                        # Format: mean ± std (mark if < 5 rounds)
                        if n_rounds == N_ROUNDS:
                            row_data[model] = f"{mean_val:.4f}±{std_val:.4f}"
                        else:
                            row_data[model] = f"{mean_val:.4f}±{std_val:.4f}*"
                    else:
                        row_data[model] = "N/A"
                
                rows.append(row_data)
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Save to CSV
        output_file = os.path.join(OUTPUT_BASE_PATH, f"summary_{metric.upper()}.csv")
        df.to_csv(output_file, index=False)
        print(f"✓ 保存到: {output_file}")
        
        # Print first few rows
        print(f"\n预览 (前10行):")
        print(df.head(10).to_string(index=False))

def is_in_distribution(test_pattern_short, train_pattern_short):
    """Determine if test pattern is in-distribution (same as training pattern)"""
    # Both should be in format like "point05", "block00", "subseq05"
    return test_pattern_short == train_pattern_short
def create_combined_table(results):
    """Create a combined table with all metrics"""
    print(f"\n{'='*80}")
    print(f"创建综合汇总表...")
    print(f"{'='*80}")
    
    rows = []
    
    for dataset, patterns in DATASET_PATTERNS.items():
        train_pattern = TRAIN_PATTERNS[dataset]
        train_short = get_pattern_short_name(train_pattern)
        
        for pattern in patterns:
            pattern_short = get_pattern_short_name(pattern)
            key = f"{dataset}_{pattern_short}"
            
            if key not in results:
                continue
            
            # Determine if in-distribution using the helper function
            is_id = is_in_distribution(pattern_short, train_short)
            
            for model in MODELS:
                if model in results[key]:
                    row_data = {
                        'Dataset': dataset,
                        'Pattern': pattern_short,
                        'Type': 'ID' if is_id else 'OOD',
                        'Model': model,
                        'MAE': f"{results[key][model]['mae_mean']:.4f}±{results[key][model]['mae_std']:.4f}",
                        'MSE': f"{results[key][model]['mse_mean']:.4f}±{results[key][model]['mse_std']:.4f}",
                        'MRE': f"{results[key][model]['mre_mean']:.4f}±{results[key][model]['mre_std']:.4f}",
                        'N_Rounds': results[key][model]['n_rounds'],
                    }
                    rows.append(row_data)
    
    df = pd.DataFrame(rows)
    output_file = os.path.join(OUTPUT_BASE_PATH, "summary_combined.csv")
    df.to_csv(output_file, index=False)
    print(f"✓ 保存到: {output_file}")
def analyze_generalization(results):
    """Analyze generalization performance (ID vs OOD)"""
    print(f"\n{'='*80}")
    print(f"泛化能力分析 (In-Distribution vs Out-of-Distribution)")
    print(f"{'='*80}\n")
    
    analysis_rows = []
    
    for dataset in DATASET_PATTERNS.keys():
        print(f"\n{dataset}:")
        print("-" * 80)
        
        train_pattern = TRAIN_PATTERNS[dataset]
        train_short = get_pattern_short_name(train_pattern)
        patterns = DATASET_PATTERNS[dataset]
        
        # Find ID pattern and OOD patterns
        id_pattern = None
        ood_patterns = []
        
        for pattern in patterns:
            pattern_short = get_pattern_short_name(pattern)
            if is_in_distribution(pattern_short, train_short):
                id_pattern = pattern
            else:
                ood_patterns.append(pattern)
        
        for model in MODELS:
            # Get ID performance
            id_mae = None
            if id_pattern:
                id_key = f"{dataset}_{get_pattern_short_name(id_pattern)}"
                if id_key in results and model in results[id_key]:
                    id_mae = results[id_key][model]['mae_mean']
            
            # Get OOD performance
            ood_maes = []
            for ood_pattern in ood_patterns:
                ood_key = f"{dataset}_{get_pattern_short_name(ood_pattern)}"
                if ood_key in results and model in results[ood_key]:
                    ood_maes.append(results[ood_key][model]['mae_mean'])
            
            if id_mae and ood_maes:
                avg_ood_mae = np.mean(ood_maes)
                degradation = ((avg_ood_mae - id_mae) / id_mae) * 100
                std_ood_mae = np.std(ood_maes)
                
                print(f"  {model:15s}: ID={id_mae:.4f}, "
                      f"OOD={avg_ood_mae:.4f}±{std_ood_mae:.4f}, "
                      f"退化={degradation:+.2f}%")
                
                analysis_rows.append({
                    'Dataset': dataset,
                    'Model': model,
                    'ID_MAE': f"{id_mae:.4f}",
                    'OOD_MAE_Mean': f"{avg_ood_mae:.4f}",
                    'OOD_MAE_Std': f"{std_ood_mae:.4f}",
                    'Degradation_%': f"{degradation:+.2f}",
                })
            elif id_mae:
                print(f"  {model:15s}: ID={id_mae:.4f}, OOD=N/A")
            else:
                print(f"  {model:15s}: 无数据")
    
    # Save generalization analysis
    if analysis_rows:
        df_analysis = pd.DataFrame(analysis_rows)
        output_file = os.path.join(OUTPUT_BASE_PATH, "generalization_analysis.csv")
        df_analysis.to_csv(output_file, index=False)
        print(f"\n✓ 泛化分析保存到: {output_file}")

def main():
    # Create output directory
    os.makedirs(OUTPUT_BASE_PATH, exist_ok=True)
    
    # Collect all results
    results = collect_results()
    
    if not results:
        print("\n[ERROR] 没有找到任何结果文件！")
        print("请确保评估任务已经完成并且结果文件存在。")
        return
    
    # Create summary tables
    create_summary_tables(results)
    
    # Create combined table
    create_combined_table(results)
    
    # Analyze generalization
    analyze_generalization(results)
    
    print(f"\n{'='*80}")
    print(f"所有结果已收集并汇总完成!")
    print(f"输出目录: {OUTPUT_BASE_PATH}/")
    print(f"  - summary_MAE.csv: MAE结果汇总")
    print(f"  - summary_MSE.csv: MSE结果汇总")
    print(f"  - summary_MRE.csv: MRE结果汇总")
    print(f"  - summary_combined.csv: 所有指标综合表")
    print(f"  - generalization_analysis.csv: 泛化能力分析")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()