#!/usr/bin/env python3
"""
Generate MSE appendix tables (same format as MAE appendix tables).
Reads from *_with_naive.csv which contain MAE, MSE, MRE columns.

Usage:
    python generate_mse_appendix.py \
      --base_path reproduce_imputation/results_csv/imputation \
      --output_dir latex_tables
"""
import os
import pandas as pd
import argparse

# ── Config (same as generate_latex_tables_v2.py) ──

EXPERIMENTS = {
    'BeijingAir': ['point01', 'point05', 'point09', 'block05', 'subseq05'],
    'ETT_h1':     ['point01', 'point05', 'point09', 'block05', 'subseq05'],
    'ItalyAir':   ['point01', 'point05', 'point09', 'block05', 'subseq05'],
    'PeMS':       ['point01', 'point05', 'point09', 'block05', 'subseq05'],
    'PhysioNet2012': ['point01'],
}

PATTERN_NAMES = {
    'point01': 'Point-10\\%',
    'point05': 'Point-50\\%',
    'point09': 'Point-90\\%',
    'block05': 'Block-50\\%',
    'subseq05': 'Subseq-50\\%',
}

DATASET_NAMES = {
    'BeijingAir': 'BeijingAir',
    'ETT_h1': 'ETT\\_h1',
    'ItalyAir': 'ItalyAir',
    'PeMS': 'PeMS',
    'PhysioNet2012': 'PhysioNet2012',
}

MODEL_ORDER = [
    'HELIX',
    'ImputeFormer', 'SAITS',
    'NonstationaryTransformer', 'PatchTST', 'iTransformer',
    'TEFN', 'TimeMixer', 'TimeMixerPP', 'ModernTCN', 'StemGNN', 'TOTEM',
    'FreTS', 'TimeLLM', 'MOMENT',
    'Naive_LinearInterp', 'Naive_LOCF', 'Naive_Median', 'Naive_Mean',
]

MODEL_DISPLAY = {
    'HELIX': 'HELIX (Ours)',
    'ImputeFormer': 'ImputeFormer',
    'SAITS': 'SAITS',
    'StemGNN': 'StemGNN',
    'Naive_LinearInterp': 'Linear Interpolation',
    'Naive_LOCF': 'LOCF',
    'Naive_Mean': 'Mean Imputation',
    'Naive_Median': 'Median Imputation',
    'PatchTST': 'PatchTST',
    'NonstationaryTransformer': 'Nonstationary Trans.',
    'FreTS': 'FreTS',
    'iTransformer': 'iTransformer',
    'TEFN': 'TEFN',
    'TimeLLM': 'Time-LLM',
    'TimeMixer': 'TimeMixer',
    'ModernTCN': 'ModernTCN',
    'TimeMixerPP': 'TimeMixer++',
    'TOTEM': 'TOTEM',
    'MOMENT': 'MOMENT',
}

ABLATION_MODELS = {
    'HELIX_NoFeatureEmbed', 'HELIX_NoFusion', 'HELIX_NoHybrid', 'HELIX_NoSinusoidalPE',
}


def format_metric(value_str):
    """Format '0.215 (0.003)' -> '0.215$\\pm$0.003'"""
    if pd.isna(value_str) or str(value_str).strip() in ('0', 'N/A', '--', ''):
        return '--'
    s = str(value_str).strip()
    if '(' in s:
        parts = s.replace(')', '').split('(')
        mean = parts[0].strip()
        std = parts[1].strip() if len(parts) > 1 else ''
        if std and std != 'N/A':
            return f"{mean}$\\pm${std}"
        return f"{mean}$\\pm$N/A"
    return s


def extract_numeric(value_str):
    if pd.isna(value_str):
        return float('inf')
    s = str(value_str).strip()
    if s in ('--', 'N/A', '', '0', 'nan', 'inf'):
        return float('inf')
    if '(' in s:
        s = s.split('(')[0].strip()
    if '$' in s:
        s = s.split('$')[0]
    try:
        v = float(s)
        return v if v > 0 else float('inf')
    except ValueError:
        return float('inf')


def get_column_ranks(values):
    indexed = [(i, extract_numeric(v)) for i, v in enumerate(values)]
    valid = [(i, v) for i, v in indexed if v != float('inf')]
    sorted_v = sorted(valid, key=lambda x: x[1])
    ranks = {}
    for rank, (idx, _) in enumerate(sorted_v, 1):
        ranks[idx] = rank
    return ranks


def format_by_rank(val, rank):
    if rank == 1:
        return f"\\textbf{{{val}}}"
    elif rank == 2:
        return f"\\underline{{{val}}}"
    return val


def generate_mse_tables(base_path, output_dir):
    """Generate one MSE appendix table per dataset."""

    all_lines = []
    all_lines.append("% Auto-generated: MSE appendix tables")
    all_lines.append("")

    for dataset, patterns in EXPERIMENTS.items():
        # Collect data
        all_data = {}
        for pattern in patterns:
            csv_path = os.path.join(base_path, pattern, f'{dataset}_with_naive.csv')
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                model = row['Model']
                if model in ABLATION_MODELS:
                    continue
                if model not in all_data:
                    all_data[model] = {}
                all_data[model][pattern] = format_metric(row.get('MSE', '--'))

        # Order models
        display_order = [m for m in MODEL_ORDER if m in all_data]
        if not display_order:
            continue

        # Column data & ranks
        columns_data = {}
        for p in patterns:
            columns_data[p] = [all_data[m].get(p, '--') for m in display_order]
        column_ranks = {p: get_column_ranks(columns_data[p]) for p in patterns}

        n_pat = len(patterns)
        col_spec = 'l|' + 'c' * n_pat
        ds_display = DATASET_NAMES.get(dataset, dataset)

        all_lines.append(r"\begin{table}[H]")
        all_lines.append(f"    \\caption{{Complete MSE results on {ds_display} across all missing patterns. Mean $\\pm$ std over 5 runs. Ranking: \\textbf{{1st}}, \\underline{{2nd}}.}}")
        all_lines.append(f"    \\label{{tab:mse_{dataset.lower()}}}")
        all_lines.append(r"    \centering")
        all_lines.append(r"    \begin{footnotesize}")
        all_lines.append(f"    \\begin{{tabular}}{{{col_spec}}}")
        all_lines.append(r"        \toprule")

        # Header
        header = ['\\textbf{Model}']
        for p in patterns:
            header.append(f"\\textbf{{{PATTERN_NAMES.get(p, p)}}}")
        all_lines.append("        " + " & ".join(header) + " \\\\")
        all_lines.append(r"        \midrule")

        # Naive section
        all_lines.append(r"        \multicolumn{" + str(n_pat + 1) + r"}{l}{\textit{Naive Baselines}} \\")
        naive_models = [m for m in display_order if m.startswith('Naive_')]
        for model in naive_models:
            idx = display_order.index(model)
            row = [MODEL_DISPLAY.get(model, model)]
            for p in patterns:
                val = columns_data[p][idx]
                rank = column_ranks[p].get(idx, 999)
                row.append(format_by_rank(val, rank) if rank <= 2 else val)
            all_lines.append("        " + " & ".join(row) + " \\\\")

        all_lines.append(r"        \midrule")
        all_lines.append(r"        \multicolumn{" + str(n_pat + 1) + r"}{l}{\textit{Deep Learning Methods}} \\")

        # DL section
        dl_models = [m for m in display_order if not m.startswith('Naive_')]
        for model in dl_models:
            idx = display_order.index(model)
            name = MODEL_DISPLAY.get(model, model)
            if model == 'HELIX':
                name = f"\\textbf{{{name}}}"
            row = [name]
            for p in patterns:
                val = columns_data[p][idx]
                rank = column_ranks[p].get(idx, 999)
                row.append(format_by_rank(val, rank) if rank <= 2 else val)
            all_lines.append("        " + " & ".join(row) + " \\\\")

        all_lines.append(r"        \bottomrule")
        all_lines.append(r"    \end{tabular}")
        all_lines.append(r"    \end{footnotesize}")
        all_lines.append(r"\end{table}")
        all_lines.append(r"\clearpage")
        all_lines.append("")

    output_path = os.path.join(output_dir, 'appendix_mse_all_datasets.tex')
    with open(output_path, 'w') as f:
        f.write('\n'.join(all_lines))
    print(f"=> Generated: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str,
                        default='reproduce_imputation/results_csv/imputation')
    parser.add_argument('--output_dir', type=str, default='./latex_tables')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Generating MSE appendix tables...")
    print(f"  base_path: {os.path.abspath(args.base_path)}")
    generate_mse_tables(args.base_path, args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()