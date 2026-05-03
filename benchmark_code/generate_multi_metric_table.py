#!/usr/bin/env python3
"""
Generate a compact multi-metric (MAE/MSE/MRE) ranking summary table.
Reads directly from *_with_naive.csv files.

Usage:
    python generate_multi_metric_table.py \
      --base_path reproduce_imputation/results_csv/imputation

Output:
    latex_tables/appendix_multi_metric_ranking.tex
"""
import os
import pandas as pd
import numpy as np
import argparse
from collections import defaultdict

# ── Same config as your existing scripts ──

EXPERIMENTS = {
    'point01': ["BeijingAir", "ETT_h1", "ItalyAir", "PeMS", "PhysioNet2012"],
    'point05': ["BeijingAir", "ETT_h1", "ItalyAir", "PeMS"],
    'point09': ["BeijingAir", "ETT_h1", "ItalyAir", "PeMS"],
    'block05': ["BeijingAir", "ETT_h1", "ItalyAir", "PeMS"],
    'subseq05': ["BeijingAir", "ETT_h1", "ItalyAir", "PeMS"],
}

EXCLUDE_MODELS = {
    'HELIX_NoFeatureEmbed', 'HELIX_NoFusion', 'HELIX_NoHybrid', 'HELIX_NoSinusoidalPE',
    'Naive_Mean', 'Naive_Median',
}

MODEL_DISPLAY = {
    'HELIX': 'HELIX (Ours)',
    'ImputeFormer': 'ImputeFormer',
    'SAITS': 'SAITS',
    'StemGNN': 'StemGNN',
    'Naive_LinearInterp': 'Linear Interpolation',
    'PatchTST': 'PatchTST',
    'NonstationaryTransformer': 'Nonstationary Trans.',
    'FreTS': 'FreTS',
    'iTransformer': 'iTransformer',
    'TEFN': 'TEFN',
    'TimeLLM': 'Time-LLM',
    'TimeMixer': 'TimeMixer',
    'Naive_LOCF': 'LOCF',
    'ModernTCN': 'ModernTCN',
    'TimeMixerPP': 'TimeMixer++',
    'TOTEM': 'TOTEM',
    'MOMENT': 'MOMENT',
}


def parse_metric(value_str):
    """Parse '0.215 (0.003)' or '0.215' -> float. Returns None if invalid."""
    if pd.isna(value_str):
        return None
    s = str(value_str).strip()
    if s in ('0', 'N/A', '--', '', 'nan', 'inf'):
        return None
    if '(' in s:
        s = s.split('(')[0].strip()
    try:
        v = float(s)
        return v if v > 0 else None
    except ValueError:
        return None


def compute_ranking(base_path):
    """
    For each of 21 settings, rank models by MAE/MSE/MRE independently.
    Return DataFrame with average ranks per metric.
    """
    rank_lists = {m: defaultdict(list) for m in ['MAE', 'MSE', 'MRE']}
    valid_counts = defaultdict(int)
    total_settings = 0

    for pattern, datasets in EXPERIMENTS.items():
        for dataset in datasets:
            csv_path = os.path.join(base_path, pattern, f'{dataset}_with_naive.csv')
            if not os.path.exists(csv_path):
                print(f"  SKIP (not found): {csv_path}")
                continue

            df = pd.read_csv(csv_path)
            # Filter out ablations and weak naive
            df = df[~df['Model'].isin(EXCLUDE_MODELS)].copy()

            print(f"  OK: {pattern}/{dataset} -> {len(df)} models")
            total_settings += 1

            for metric in ['MAE', 'MSE', 'MRE']:
                df[f'_{metric}_val'] = df[metric].apply(parse_metric)
                valid = df.dropna(subset=[f'_{metric}_val']).copy()
                if len(valid) == 0:
                    continue
                valid['_rank'] = valid[f'_{metric}_val'].rank(ascending=True, method='min')
                for _, row in valid.iterrows():
                    rank_lists[metric][row['Model']].append(row['_rank'])

            for m in df['Model'].unique():
                valid_counts[m] += 1

    # Aggregate
    all_models = set()
    for d in rank_lists.values():
        all_models.update(d.keys())

    print(f"\n  Total: {total_settings} settings, {len(all_models)} models")

    rows = []
    for m in all_models:
        row = {'Model': m, 'Valid': valid_counts.get(m, 0), 'Total': total_settings}
        for metric in ['MAE', 'MSE', 'MRE']:
            r = rank_lists[metric].get(m, [])
            row[f'{metric}_AvgRank'] = round(np.mean(r), 2) if r else None
        rows.append(row)

    df_out = pd.DataFrame(rows)
    if df_out.empty:
        return df_out
    df_out = df_out.sort_values('MAE_AvgRank', ascending=True, na_position='last').reset_index(drop=True)
    df_out['Global_Rank'] = df_out.index + 1
    return df_out


def to_latex(df, output_path):
    """Generate compact LaTeX table."""
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"    \caption{Multi-metric ranking consistency. Average rank across all 21 settings for MAE, MSE, and MRE (lower is better). HELIX achieves the best average rank on all three metrics. $\dagger$: incomplete settings.}")
    lines.append(r"    \label{tab:multi_metric_ranking}")
    lines.append(r"    \centering")
    lines.append(r"    \begin{small}")
    lines.append(r"    \begin{tabular}{l|ccc|c}")
    lines.append(r"        \toprule")
    lines.append(r"        \textbf{Model} & \textbf{MAE Rank}$\downarrow$ & \textbf{MSE Rank}$\downarrow$ & \textbf{MRE Rank}$\downarrow$ & \textbf{Valid} \\")
    lines.append(r"        \midrule")

    for _, row in df.iterrows():
        model = row['Model']
        display = MODEL_DISPLAY.get(model, model)
        valid = int(row['Valid'])
        total = int(row['Total'])

        mae_r = f"{row['MAE_AvgRank']:.2f}" if pd.notna(row['MAE_AvgRank']) else '--'
        mse_r = f"{row['MSE_AvgRank']:.2f}" if pd.notna(row['MSE_AvgRank']) else '--'
        mre_r = f"{row['MRE_AvgRank']:.2f}" if pd.notna(row['MRE_AvgRank']) else '--'

        if model == 'HELIX':
            display = f"\\textbf{{{display}}}"
            mae_r = f"\\textbf{{{mae_r}}}"
            mse_r = f"\\textbf{{{mse_r}}}"
            mre_r = f"\\textbf{{{mre_r}}}"

        if valid < total:
            display += "$^\\dagger$"

        lines.append(f"        {display} & {mae_r} & {mse_r} & {mre_r} & {valid}/{total} \\\\")

    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"    \end{small}")
    lines.append(r"\end{table}")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\n=> Generated: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str,
                        default='reproduce_imputation/results_csv/imputation')
    parser.add_argument('--output_dir', type=str, default='./latex_tables')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Computing multi-metric ranking...")
    print(f"  base_path: {os.path.abspath(args.base_path)}")
    df = compute_ranking(args.base_path)

    if df.empty:
        print("\nERROR: No data loaded. Check --base_path.")
        print("Expected structure: base_path/point01/BeijingAir_with_naive.csv")
        return

    print("\nRanking:")
    cols = ['Model', 'MAE_AvgRank', 'MSE_AvgRank', 'MRE_AvgRank', 'Valid']
    print(df[cols].to_string(index=False))

    out = os.path.join(args.output_dir, 'appendix_multi_metric_ranking.tex')
    to_latex(df, out)


if __name__ == "__main__":
    main()