"""
Parse downstream task results and generate summary tables.
Output format matches the original benchmark code style.
"""

import os
import re
import argparse
import pandas as pd
import numpy as np
from collections import defaultdict


# Model categories for ranking
MODEL_CATEGORIES = {
    'HELIX': 'Ours',
    'HELIX_NoFeatureEmbed': 'Ablation',
    'HELIX_NoFusion': 'Ablation',
    'HELIX_NoHybrid': 'Ablation',
    'HELIX_NoRotaryPE': 'Ablation',
    'TEFN': 'Recent (2024)',
    'TimeMixerPP': 'Recent (2024)',
    'TimeLLM': 'Foundation',
    'MOMENT': 'Foundation',
    'TimeMixer': 'Recent (2024)',
    'ModernTCN': 'Recent (2024)',
    'ImputeFormer': 'Imputation-specific',
    'TOTEM': 'Recent (2024)',
    'iTransformer': 'Transformer',
    'SAITS': 'Imputation-specific',
    'FreTS': 'Frequency-domain',
    'NonstationaryTransformer': 'Transformer',
    'PatchTST': 'Transformer',
    'Naive_mean': 'Naive',
    'Naive_median': 'Naive',
    'Naive_locf': 'Naive',
    'Naive_linear_interpolation': 'Naive',
}


def parse_classification_log(log_path):
    """Parse classification log file and extract metrics."""
    results = {}
    
    if not os.path.exists(log_path):
        return None
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # 方式1: 尝试解析标准格式 "=== Average Results ==="
    avg_section = re.search(r'=== Average Results ===(.*)', content, re.DOTALL)
    if avg_section:
        avg_content = avg_section.group(1)
        patterns = {
            'XGB_ROC_AUC': r'XGB_ROC_AUC:\s*([\d.]+)',
            'XGB_PR_AUC': r'XGB_PR_AUC:\s*([\d.]+)',
            'RNN_ROC_AUC': r'RNN_ROC_AUC:\s*([\d.]+)',
            'RNN_PR_AUC': r'RNN_PR_AUC:\s*([\d.]+)',
            'Transformer_ROC_AUC': r'Transformer_ROC_AUC:\s*([\d.]+)',
            'Transformer_PR_AUC': r'Transformer_PR_AUC:\s*([\d.]+)',
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, avg_content)
            if match:
                results[key] = float(match.group(1))
        if results:
            return results
    
    # 方式2: 尝试解析HELIX格式 "XGB with XXX imputation PR_AUC: x.xxxx±x.xxxx, ROC_AUC: x.xxxx±x.xxxx"
    helix_patterns = {
        ('XGB_PR_AUC', 'XGB_ROC_AUC'): r'XGB with \w+ imputation PR_AUC:\s*([\d.]+)[±\+\-][\d.]+,\s*ROC_AUC:\s*([\d.]+)',
        ('RNN_PR_AUC', 'RNN_ROC_AUC'): r'RNN with \w+ imputation PR_AUC:\s*([\d.]+)[±\+\-][\d.]+,\s*ROC_AUC:\s*([\d.]+)',
        ('Transformer_PR_AUC', 'Transformer_ROC_AUC'): r'Transformer with \w+ imputation PR_AUC:\s*([\d.]+)[±\+\-][\d.]+,\s*ROC_AUC:\s*([\d.]+)',
    }
    for (pr_key, roc_key), pattern in helix_patterns.items():
        match = re.search(pattern, content)
        if match:
            results[pr_key] = float(match.group(1))
            results[roc_key] = float(match.group(2))
    
    return results if results else None


def parse_forecasting_log(log_path):
    """Parse forecasting log file and extract metrics."""
    results = {}
    
    if not os.path.exists(log_path):
        return None
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # 方式1: 尝试解析标准格式 "=== Average Results ==="
    avg_section = re.search(r'=== Average Results ===(.*)', content, re.DOTALL)
    if avg_section:
        avg_content = avg_section.group(1)
        patterns = {
            'XGB_MAE': r'XGB_MAE:\s*([\d.]+)',
            'XGB_MSE': r'XGB_MSE:\s*([\d.]+)',
            'XGB_MRE': r'XGB_MRE:\s*([\d.]+)',
            'RNN_MAE': r'RNN_MAE:\s*([\d.]+)',
            'RNN_MSE': r'RNN_MSE:\s*([\d.]+)',
            'RNN_MRE': r'RNN_MRE:\s*([\d.]+)',
            'Transformer_MAE': r'Transformer_MAE:\s*([\d.]+)',
            'Transformer_MSE': r'Transformer_MSE:\s*([\d.]+)',
            'Transformer_MRE': r'Transformer_MRE:\s*([\d.]+)',
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, avg_content)
            if match:
                results[key] = float(match.group(1))
        if results:
            return results
    
    # 方式2: 尝试解析HELIX格式 "XGB (with XXX imputation) regression MAE: x.xxxx±x.xxxx, MSE: x.xxxx±x.xxxx, MRE: x.xxxx±x.xxxx"
    helix_patterns = {
        ('XGB_MAE', 'XGB_MSE', 'XGB_MRE'): r'XGB \(with \w+ imputation\) (?:regression|forecasting) MAE:\s*([\d.]+)[±\+\-][\d.]+,\s*MSE:\s*([\d.]+)[±\+\-][\d.]+,\s*MRE:\s*([\d.]+)',
        ('RNN_MAE', 'RNN_MSE', 'RNN_MRE'): r'RNN \(with \w+ imputation\) (?:regression|forecasting) MAE:\s*([\d.]+)[±\+\-][\d.]+,\s*MSE:\s*([\d.]+)[±\+\-][\d.]+,\s*MRE:\s*([\d.]+)',
        ('Transformer_MAE', 'Transformer_MSE', 'Transformer_MRE'): r'Transformer \(with \w+ imputation\) (?:regression|forecasting) MAE:\s*([\d.]+)[±\+\-][\d.]+,\s*MSE:\s*([\d.]+)[±\+\-][\d.]+,\s*MRE:\s*([\d.]+)',
    }
    for (mae_key, mse_key, mre_key), pattern in helix_patterns.items():
        match = re.search(pattern, content)
        if match:
            results[mae_key] = float(match.group(1))
            results[mse_key] = float(match.group(2))
            results[mre_key] = float(match.group(3))
    
    return results if results else None


def parse_regression_log(log_path):
    """Parse regression log file and extract metrics."""
    # Same format as forecasting
    return parse_forecasting_log(log_path)


def compute_rankings(df, metric_cols, higher_is_better=True):
    """Compute rankings for each metric and average rank."""
    rankings = pd.DataFrame(index=df.index)
    
    for col in metric_cols:
        if col in df.columns:
            if higher_is_better:
                rankings[f'{col}_rank'] = df[col].rank(ascending=False, method='min')
            else:
                rankings[f'{col}_rank'] = df[col].rank(ascending=True, method='min')
    
    rank_cols = [c for c in rankings.columns if c.endswith('_rank')]
    if rank_cols:
        rankings['Avg_Rank'] = rankings[rank_cols].mean(axis=1)
    
    return rankings


def parse_all_results(output_dir):
    """Parse all log files and generate summary tables."""
    
    all_models = list(MODEL_CATEGORIES.keys())
    
    # ==========================================
    # Classification Results
    # ==========================================
    print("Parsing classification results...")
    classification_results = {}
    log_dir = os.path.join(output_dir, 'classification', 'logs')
    
    for model in all_models:
        log_path = os.path.join(log_dir, f'{model}.log')
        results = parse_classification_log(log_path)
        if results:
            classification_results[model] = results
    
    if classification_results:
        clf_df = pd.DataFrame(classification_results).T
        clf_df.index.name = 'Model'
        
        # Reorder columns
        clf_cols = ['XGB_PR_AUC', 'XGB_ROC_AUC', 'RNN_PR_AUC', 'RNN_ROC_AUC', 
                    'Transformer_PR_AUC', 'Transformer_ROC_AUC']
        clf_df = clf_df[[c for c in clf_cols if c in clf_df.columns]]
        
        # Compute rankings (higher is better for AUC)
        clf_rankings = compute_rankings(clf_df, clf_df.columns, higher_is_better=True)
        clf_df['Avg_Rank'] = clf_rankings['Avg_Rank']
        clf_df['Category'] = clf_df.index.map(MODEL_CATEGORIES)
        
        # Sort by average rank
        clf_df = clf_df.sort_values('Avg_Rank')
        clf_df['Global_Rank'] = range(1, len(clf_df) + 1)
        
        # Save
        clf_df.to_csv(os.path.join(output_dir, 'classification', 'PhysioNet2012_point01_classification.csv'))
        print(f"  Saved: {os.path.join(output_dir, 'classification', 'PhysioNet2012_point01_classification.csv')}")
        print(f"  Found {len(clf_df)} models")
    
    # ==========================================
    # Forecasting Results
    # ==========================================
    print("Parsing forecasting results...")
    forecasting_results = {}
    log_dir = os.path.join(output_dir, 'forecasting', 'logs')
    
    for model in all_models:
        log_path = os.path.join(log_dir, f'{model}.log')
        results = parse_forecasting_log(log_path)
        if results:
            forecasting_results[model] = results
    
    if forecasting_results:
        fore_df = pd.DataFrame(forecasting_results).T
        fore_df.index.name = 'Model'
        
        # Reorder columns
        fore_cols = ['XGB_MAE', 'XGB_MSE', 'XGB_MRE', 'RNN_MAE', 'RNN_MSE', 'RNN_MRE',
                     'Transformer_MAE', 'Transformer_MSE', 'Transformer_MRE']
        fore_df = fore_df[[c for c in fore_cols if c in fore_df.columns]]
        
        # Compute rankings (lower is better for error metrics)
        fore_rankings = compute_rankings(fore_df, fore_df.columns, higher_is_better=False)
        fore_df['Avg_Rank'] = fore_rankings['Avg_Rank']
        fore_df['Category'] = fore_df.index.map(MODEL_CATEGORIES)
        
        fore_df = fore_df.sort_values('Avg_Rank')
        fore_df['Global_Rank'] = range(1, len(fore_df) + 1)
        
        fore_df.to_csv(os.path.join(output_dir, 'forecasting', 'ETT_h1_block05_forecasting.csv'))
        print(f"  Saved: {os.path.join(output_dir, 'forecasting', 'ETT_h1_block05_forecasting.csv')}")
        print(f"  Found {len(fore_df)} models")
    
    # ==========================================
    # Regression Results
    # ==========================================
    print("Parsing regression results...")
    regression_results = {}
    log_dir = os.path.join(output_dir, 'regression', 'logs')
    
    for model in all_models:
        log_path = os.path.join(log_dir, f'{model}.log')
        results = parse_regression_log(log_path)
        if results:
            regression_results[model] = results
    
    if regression_results:
        reg_df = pd.DataFrame(regression_results).T
        reg_df.index.name = 'Model'
        
        reg_cols = ['XGB_MAE', 'XGB_MSE', 'XGB_MRE', 'RNN_MAE', 'RNN_MSE', 'RNN_MRE',
                    'Transformer_MAE', 'Transformer_MSE', 'Transformer_MRE']
        reg_df = reg_df[[c for c in reg_cols if c in reg_df.columns]]
        
        reg_rankings = compute_rankings(reg_df, reg_df.columns, higher_is_better=False)
        reg_df['Avg_Rank'] = reg_rankings['Avg_Rank']
        reg_df['Category'] = reg_df.index.map(MODEL_CATEGORIES)
        
        reg_df = reg_df.sort_values('Avg_Rank')
        reg_df['Global_Rank'] = range(1, len(reg_df) + 1)
        
        reg_df.to_csv(os.path.join(output_dir, 'regression', 'ETT_h1_block05_regression.csv'))
        print(f"  Saved: {os.path.join(output_dir, 'regression', 'ETT_h1_block05_regression.csv')}")
        print(f"  Found {len(reg_df)} models")
    
    # ==========================================
    # Generate Overall Summary
    # ==========================================
    print("\nGenerating overall summary...")
    
    summary_data = []
    
    for model in all_models:
        row = {'Model': model, 'Category': MODEL_CATEGORIES.get(model, 'Unknown')}
        
        # Get ranks from each task
        ranks = []
        
        if classification_results and model in classification_results:
            clf_idx = clf_df.index.tolist().index(model) if model in clf_df.index else None
            if clf_idx is not None:
                row['Classification_Rank'] = clf_df.loc[model, 'Avg_Rank']
                ranks.append(clf_df.loc[model, 'Avg_Rank'])
        
        if forecasting_results and model in forecasting_results:
            if model in fore_df.index:
                row['Forecasting_Rank'] = fore_df.loc[model, 'Avg_Rank']
                ranks.append(fore_df.loc[model, 'Avg_Rank'])
        
        if regression_results and model in regression_results:
            if model in reg_df.index:
                row['Regression_Rank'] = reg_df.loc[model, 'Avg_Rank']
                ranks.append(reg_df.loc[model, 'Avg_Rank'])
        
        if ranks:
            row['Overall_Avg_Rank'] = np.mean(ranks)
            row['Valid_Tasks'] = len(ranks)
            summary_data.append(row)
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Overall_Avg_Rank')
        summary_df['Global_Rank'] = range(1, len(summary_df) + 1)
        
        # Reorder columns
        col_order = ['Model', 'Overall_Avg_Rank', 'Classification_Rank', 'Forecasting_Rank', 
                     'Regression_Rank', 'Valid_Tasks', 'Category', 'Global_Rank']
        summary_df = summary_df[[c for c in col_order if c in summary_df.columns]]
        
        summary_df.to_csv(os.path.join(output_dir, 'downstream_summary.csv'), index=False)
        print(f"Saved: {os.path.join(output_dir, 'downstream_summary.csv')}")
        
        # Print summary table
        print("\n" + "=" * 80)
        print("DOWNSTREAM TASK SUMMARY")
        print("=" * 80)
        print(summary_df.to_string(index=False))
        print("=" * 80)
    
    return classification_results, forecasting_results, regression_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, 
                        default="/home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/downstream_result")
    args = parser.parse_args()
    
    parse_all_results(args.output_dir)


if __name__ == "__main__":
    main()