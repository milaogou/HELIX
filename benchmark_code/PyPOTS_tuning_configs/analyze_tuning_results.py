# -*- coding: utf-8 -*-
"""
收集和分析超参数调优结果
"""
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

TUNING_OUTPUT_PATH = "hyperparameter_tuning_results"

def load_metrics(trial_dir: str) -> dict:
    """从metrics.json加载指标"""
    metrics_file = os.path.join(trial_dir, "metrics.json")
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}


def check_training_failed(trial_dir: str) -> tuple:
    """检查训练是否失败以及失败原因"""
    failed_file = os.path.join(trial_dir, "training_failed.txt")
    if os.path.exists(failed_file):
        try:
            with open(failed_file, 'r') as f:
                reason = f.read().strip()
                return True, reason
        except:
            return True, "Unknown error"
    return False, ""


def collect_tuning_results(tuning_dir: str) -> pd.DataFrame:
    """收集某个模型-数据集组合的所有trial结果"""
    results = []
    
    tuning_path = Path(tuning_dir)
    if not tuning_path.exists():
        print(f"目录不存在: {tuning_dir}")
        return pd.DataFrame()
    
    # 查找所有trial
    for params_file in tuning_path.glob("trial_*_params.json"):
        trial_id = params_file.stem.split('_')[1]
        
        # 读取参数
        try:
            with open(params_file, 'r') as f:
                params = json.load(f)
        except:
            continue
        
        # 读取指标
        trial_dir = tuning_path / f"trial_{trial_id}"
        metrics = load_metrics(str(trial_dir))
        
        # 检查是否完成
        status_file = tuning_path / f"trial_{trial_id}_status.txt"
        completed = status_file.exists()
        
        # 检查是否失败
        failed, fail_reason = check_training_failed(str(trial_dir))
        
        result = {
            'trial_id': int(trial_id),
            'completed': completed,
            'failed': failed,
            'fail_reason': fail_reason if failed else "",
            **params,
            **metrics
        }
        results.append(result)
    
    df = pd.DataFrame(results)
    return df


def analyze_model_tuning(model_name: str, dataset_name: str, metric='mae'):
    """分析单个模型的调优结果"""
    tuning_dir = os.path.join(TUNING_OUTPUT_PATH, f"{model_name}_{dataset_name}_tuning")
    
    print(f"\n{'='*70}")
    print(f"分析 {model_name} 在 {dataset_name} 数据集上的调优结果")
    print(f"{'='*70}")
    
    df = collect_tuning_results(tuning_dir)
    
    if df.empty:
        print(f"⚠️  没有找到结果")
        return None, {'total': 0, 'completed': 0, 'failed': 0, 'valid': 0}
    
    # 统计信息
    total_trials = len(df)
    completed_trials = df['completed'].sum()
    failed_trials = df['failed'].sum() if 'failed' in df.columns else 0
    
    print(f"\n统计信息:")
    print(f"  总trials数: {total_trials}")
    print(f"  已完成: {completed_trials}")
    print(f"  失败: {failed_trials}")
    
    # 检查metric列是否存在
    if metric not in df.columns:
        print(f"⚠️  没有找到指标 '{metric}'")
        print(f"  可用列: {list(df.columns)}")
        
        # 保存失败信息
        failed_df = df[df['completed'] == False]
        if not failed_df.empty:
            failed_path = os.path.join(tuning_dir, "failed_trials.csv")
            failed_df.to_csv(failed_path, index=False)
            print(f"\n✓ 失败trials信息已保存到: {failed_path}")
        
        stats = {
            'total': total_trials,
            'completed': completed_trials,
            'failed': failed_trials,
            'valid': 0
        }
        return None, stats
    
    # 过滤出已完成且有指标的trials
    df_valid = df[df['completed'] & df[metric].notna() & (~df['failed'])].copy()
    valid_trials = len(df_valid)
    
    print(f"  有效结果: {valid_trials}")
    
    if df_valid.empty:
        print(f"⚠️  没有有效结果")
        
        # 分析失败原因
        if 'fail_reason' in df.columns:
            failed_df = df[df['failed']]
            if not failed_df.empty:
                print(f"\n失败原因统计:")
                fail_reasons = failed_df['fail_reason'].value_counts()
                for reason, count in fail_reasons.items():
                    print(f"  {reason[:50]}: {count} trials")
        
        # 保存失败信息
        failed_df = df[~df['completed'] | df['failed']]
        if not failed_df.empty:
            failed_path = os.path.join(tuning_dir, "failed_trials.csv")
            failed_df.to_csv(failed_path, index=False)
            print(f"\n✓ 失败trials信息已保存到: {failed_path}")
        
        stats = {
            'total': total_trials,
            'completed': completed_trials,
            'failed': failed_trials,
            'valid': valid_trials
        }
        return None, stats
    
    # 排序找最佳
    df_sorted = df_valid.sort_values(metric)
    
    print(f"\n性能分布:")
    print(f"  最佳 {metric.upper()}: {df_sorted[metric].iloc[0]:.6f}")
    print(f"  最差 {metric.upper()}: {df_sorted[metric].iloc[-1]:.6f}")
    print(f"  平均 {metric.upper()}: {df_sorted[metric].mean():.6f}")
    print(f"  标准差: {df_sorted[metric].std():.6f}")
    
    # Top 5配置
    print(f"\nTop 5 最佳配置:")
    print("="*70)
    for idx, row in df_sorted.head(5).iterrows():
        print(f"\n[Rank {list(df_sorted.index).index(idx) + 1}] Trial {int(row['trial_id'])}")
        print(f"  {metric.upper()}: {row[metric]:.6f}")
        
        # 打印关键参数
        exclude_params = ['trial_id', 'completed', 'failed', 'fail_reason',
                         'mae', 'mse', 'mre', 'inference_time',
                         'n_steps', 'n_features', 'epochs', 'patience',
                         'ORT_weight', 'MIT_weight', 'input_dim', 'output_dim']
        print(f"  关键参数:")
        for key, value in row.items():
            if key not in exclude_params and pd.notna(value):
                print(f"    {key}: {value}")
    
    # 保存完整结果
    output_path = os.path.join(tuning_dir, f"tuning_results_summary.csv")
    df_sorted.to_csv(output_path, index=False)
    print(f"\n✓ 完整结果已保存到: {output_path}")
    
    # 保存最佳配置
    best_config = df_sorted.iloc[0].to_dict()
    # 移除非配置项
    for key in ['trial_id', 'completed', 'failed', 'fail_reason', 'mae', 'mse', 'mre', 'inference_time']:
        best_config.pop(key, None)
    
    best_config_path = os.path.join(tuning_dir, f"best_config.json")
    with open(best_config_path, 'w') as f:
        json.dump(best_config, f, indent=2)
    print(f"✓ 最佳配置已保存到: {best_config_path}")
    
    # 保存失败的trials信息
    failed_df = df[~df['completed'] | df['failed']]
    if not failed_df.empty:
        failed_path = os.path.join(tuning_dir, "failed_trials.csv")
        failed_df.to_csv(failed_path, index=False)
        print(f"✓ 失败trials信息已保存到: {failed_path}")
    
    stats = {
        'total': total_trials,
        'completed': completed_trials,
        'failed': failed_trials,
        'valid': valid_trials
    }
    
    return df_sorted, stats


def analyze_all_tuning_results():
    """分析所有调优结果"""
    base_path = Path(TUNING_OUTPUT_PATH)
    
    if not base_path.exists():
        print(f"结果目录不存在: {TUNING_OUTPUT_PATH}")
        return
    
    tuning_dirs = [d for d in base_path.iterdir() if d.is_dir() and '_tuning' in d.name]
    
    print(f"找到 {len(tuning_dirs)} 个调优结果目录")
    
    summary_results = []
    
    for tuning_dir in sorted(tuning_dirs):
        dir_name = tuning_dir.name.replace('_tuning', '')
        parts = dir_name.rsplit('_', 1)
        if len(parts) == 2:
            model_name, dataset_name = parts
        else:
            continue
        
        try:
            df, stats = analyze_model_tuning(model_name, dataset_name)
            
            result = {
                'model': model_name,
                'dataset': dataset_name,
                'total_trials': stats['total'],
                'completed': stats['completed'],
                'failed': stats['failed'],
                'valid': stats['valid'],
                'success_rate': f"{stats['valid']/stats['total']*100:.1f}%" if stats['total'] > 0 else "0%",
            }
            
            if df is not None and not df.empty:
                best_result = df.iloc[0]
                result.update({
                    'best_mae': best_result.get('mae', np.nan),
                    'best_mse': best_result.get('mse', np.nan),
                    'best_mre': best_result.get('mre', np.nan),
                })
            else:
                result.update({
                    'best_mae': np.nan,
                    'best_mse': np.nan,
                    'best_mre': np.nan,
                })
            
            summary_results.append(result)
            
        except Exception as e:
            print(f"\n⚠️  处理 {model_name}_{dataset_name} 时出错: {e}")
            summary_results.append({
                'model': model_name,
                'dataset': dataset_name,
                'total_trials': 0,
                'completed': 0,
                'failed': 0,
                'valid': 0,
                'success_rate': "0%",
                'best_mae': np.nan,
                'best_mse': np.nan,
                'best_mre': np.nan,
            })
    
    if summary_results:
        summary_df = pd.DataFrame(summary_results)
        summary_path = os.path.join(TUNING_OUTPUT_PATH, "tuning_summary_all.csv")
        summary_df.to_csv(summary_path, index=False)
        
        print(f"\n{'='*70}")
        print("所有模型调优结果总结:")
        print(f"{'='*70}")
        print(summary_df.to_string(index=False))
        print(f"\n✓ 总结报告已保存到: {summary_path}")
        
        # 打印失败统计
        print(f"\n{'='*70}")
        print("失败统计:")
        print(f"{'='*70}")
        total_all = summary_df['total_trials'].sum()
        failed_all = summary_df['failed'].sum()
        valid_all = summary_df['valid'].sum()
        print(f"总trials: {total_all}")
        print(f"成功: {valid_all} ({valid_all/total_all*100:.1f}%)")
        print(f"失败: {failed_all} ({failed_all/total_all*100:.1f}%)")
        
        # 显示哪些模型有问题
        problem_models = summary_df[summary_df['valid'] == 0]
        if not problem_models.empty:
            print(f"\n以下模型-数据集组合没有有效结果:")
            for _, row in problem_models.iterrows():
                print(f"  - {row['model']} on {row['dataset']}: {row['failed']}/{row['total_trials']} 失败")


if __name__ == "__main__":
    analyze_all_tuning_results()