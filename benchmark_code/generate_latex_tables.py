#!/usr/bin/env python3
"""
Generate LaTeX tables from CSV results for HELIX ICML 2026 paper.
Usage:
    python generate_latex_tables_v2.py --base_path /path/to/results_csv/imputation
Output:
    - table1_overall_ranking.tex
    - table2_detailed_results.tex  
    - table3_ablation.tex
    - appendix_*.tex (6 files, one per dataset)
"""
import os
import pandas as pd
import argparse
from collections import defaultdict

# =============================================================================
# Configuration
# =============================================================================

# =============================================================================
# Main comparison set (A-method): HELIX + baselines only (exclude Ablation + Naive)
# =============================================================================

EXCLUDE_CATEGORIES_FOR_MAIN = {'Ablation', 'Naive'}
# Only include these two naive baselines in Table 1 for reference
TABLE1_ALLOW_NAIVE = {'Naive_LOCF', 'Naive_LinearInterp'}

def is_main_model(model: str) -> bool:
    """True if model participates in main ranking/comparison (Table 1)."""
    if model == 'HELIX':
        return True

    # Special-case: allow two strong naive baselines for Table 1 comparison
    if model in TABLE1_ALLOW_NAIVE:
        return True

    cat = get_category(model)  # from MODEL_DISPLAY second field
    return cat not in EXCLUDE_CATEGORIES_FOR_MAIN



# Model display names and categories
MODEL_DISPLAY = {
    'HELIX': ('HELIX (Ours)', 'Ours'),
    'HELIX_NoFeatureEmbed': ('w/o Feature Identity Emb.', 'Ablation'),
    'HELIX_NoFusion': ('w/o Multi-level Fusion', 'Ablation'),
    'HELIX_NoHybrid': ('w/o Hybrid Encoding', 'Ablation'),
    'HELIX_NoRotaryPE': ('w/o Rotary PE', 'Ablation'),
    'ImputeFormer': ('ImputeFormer', 'Imputation-specific'),
    'SAITS': ('SAITS', 'Imputation-specific'),
    'TEFN': ('TEFN', 'Recent (2024)'),
    'TimeMixer': ('TimeMixer', 'Recent (2024)'),
    'TimeMixerPP': ('TimeMixer++', 'Recent (2024)'),
    'ModernTCN': ('ModernTCN', 'Recent (2024)'),
    'StemGNN': ('StemGNN', 'Graph Neural Network'),
    'TOTEM': ('TOTEM', 'Recent (2024)'),
    'iTransformer': ('iTransformer', 'Transformer'),
    'NonstationaryTransformer': ('Nonstationary Trans.', 'Transformer'),
    'PatchTST': ('PatchTST', 'Transformer'),
    'FreTS': ('FreTS', 'Frequency-domain'),
    'TimeLLM': ('Time-LLM', 'Foundation'),
    'MOMENT': ('MOMENT', 'Foundation'),
    'Naive_Mean': ('Mean Imputation', 'Naive'),
    'Naive_Median': ('Median Imputation', 'Naive'),
    'Naive_LOCF': ('LOCF', 'Naive'),
    'Naive_LinearInterp': ('Linear Interpolation', 'Naive'),
}
# Model categories (fine-grained)
MODEL_CATEGORY = {
    'HELIX': 'Ours',
    'HELIX_NoFusion': 'Ablation',
    'HELIX_NoRotaryPE': 'Ablation',
    'HELIX_NoHybrid': 'Ablation',
    'HELIX_NoFeatureEmbed': 'Ablation',
    'ImputeFormer': 'Low-rank Attention',
    'SAITS': 'Masked Attention',
    'PatchTST': 'Patch-based',
    'FreTS': 'Frequency Domain',
    'NonstationaryTransformer': 'Non-stationary Attn',
    'iTransformer': 'Variate Attention',
    'TEFN': 'Evidence Fusion',
    'TimeLLM': 'LLM Adaptation',
    'TimeMixer': 'Multi-scale Mixing',
    'TimeMixerPP': 'Multi-scale Mixing',
    'ModernTCN': 'Modern Convolution',
    'TOTEM': 'Tokenization',
    'MOMENT': 'Foundation Model',
    'StemGNN': 'Graph Neural Network',
    'Naive_Mean': 'Naive',
    'Naive_Median': 'Naive',
    'Naive_LOCF': 'Naive',
    'Naive_LinearInterp': 'Naive',
}

# Model publication venues
MODEL_VENUE = {
    'HELIX': '--',
    'HELIX_NoFusion': '--',
    'HELIX_NoRotaryPE': '--',
    'HELIX_NoHybrid': '--',
    'HELIX_NoFeatureEmbed': '--',
    'ImputeFormer': "KDD'24",
    'SAITS': "ESWA'23",
    'PatchTST': "ICLR'23",
    'FreTS': "NeurIPS'23",
    'NonstationaryTransformer': "NeurIPS'22",
    'iTransformer': "ICLR'24",
    'TEFN': "TPAMI'25",
    'TimeLLM': "ICLR'24",
    'TimeMixer': "ICLR'24",
    'TimeMixerPP': "ICLR'25",
    'ModernTCN': "ICLR'24",
    'TOTEM': "TMLR'24",
    'MOMENT': "ICML'24",
    'StemGNN': "NeurIPS'20",
    'Naive_Mean': '--',
    'Naive_Median': '--',
    'Naive_LOCF': '--',
    'Naive_LinearInterp': '--',
}
# Order for display (HELIX first, then ablations, then baselines by category)
MODEL_ORDER = [
    'HELIX',
    'HELIX_NoFusion', 'HELIX_NoRotaryPE', 'HELIX_NoHybrid', 'HELIX_NoFeatureEmbed',
    'ImputeFormer', 'SAITS',
    'NonstationaryTransformer', 'PatchTST', 'iTransformer',
    'TEFN', 'TimeMixer', 'TimeMixerPP', 'ModernTCN', 'StemGNN', 'TOTEM',
    'FreTS',
    'TimeLLM', 'MOMENT',
    'Naive_LinearInterp', 'Naive_LOCF', 'Naive_Median', 'Naive_Mean',
]

DATASET_NAMES = {
    'BeijingAir': 'BeijingAir (24 steps, 132 features)',
    'ETT_h1': 'ETT-h1 (48 steps, 7 features)',
    'ItalyAir': 'ItalyAir (12 steps, 13 features)',
    'PeMS': 'PeMS (24 steps, 862 features)',
    'PhysioNet2012': 'PhysioNet2012 (48 steps, 35 features)',
}

PATTERN_NAMES = {
    'point01': 'Point-10\\%',
    'point05': 'Point-50\\%',
    'point09': 'Point-90\\%',
    'block05': 'Block-50\\%',
    'subseq05': 'Subseq-50\\%',
}

# =============================================================================
# Helper Functions
# =============================================================================

def compute_main_global_ranking_from_setting_csvs(base_path):
    """
    Compute global ranking (Avg_Rank, Global_Rank, Valid_Experiments, Total_Experiments)
    over ALL dataset x pattern settings, ranking among main models only:
    HELIX + baselines (exclude Ablation + Naive).
    """
    datasets_patterns = {
        'BeijingAir': ['point01', 'point05', 'point09', 'block05', 'subseq05'],
        'ETT_h1': ['point01', 'point05', 'point09', 'block05', 'subseq05'],
        'ItalyAir': ['point01', 'point05', 'point09', 'block05', 'subseq05'],
        'PeMS': ['point01', 'point05', 'point09', 'block05', 'subseq05'],
        'PhysioNet2012': ['point01'],
    }

    # Collect per-model ranks across settings
    rank_lists = defaultdict(list)
    valid_counts = defaultdict(int)
    total_settings = 0

    for dataset, patterns in datasets_patterns.items():
        for pattern in patterns:
            csv_path = os.path.join(base_path, pattern, f'{dataset}_with_naive.csv')
            if not os.path.exists(csv_path):
                continue

            df = pd.read_csv(csv_path)

            # Filter to main models only
            df_main = df[df['Model'].apply(is_main_model)].copy()
            if len(df_main) == 0:
                continue

            total_settings += 1

            # Extract numeric MAE (lower is better)
            maes = []
            models = []
            for _, row in df_main.iterrows():
                models.append(row['Model'])
                maes.append(extract_numeric_value(row['MAE']))

            # Rank within this setting (skip inf)
            indexed = [(i, maes[i]) for i in range(len(maes)) if maes[i] != float('inf')]
            indexed_sorted = sorted(indexed, key=lambda x: x[1])

            # Assign 1..K ranks (no ties handling here; if you need ties, add tie-aware ranking)
            setting_ranks = {}
            for r, (i, _) in enumerate(indexed_sorted, start=1):
                setting_ranks[models[i]] = r

            # Append
            for m in models:
                if m in setting_ranks:
                    rank_lists[m].append(setting_ranks[m])
                    valid_counts[m] += 1
                else:
                    # missing / invalid MAE in this setting
                    pass

    # Build summary df
    rows = []
    for m, ranks in rank_lists.items():
        if len(ranks) == 0:
            continue
        avg_rank = sum(ranks) / len(ranks)
        rows.append({
            'Model': m,
            'Avg_Rank': avg_rank,
            'Valid_Experiments': valid_counts[m],
            'Total_Experiments': total_settings,
        })

    out = pd.DataFrame(rows)
    out = out.sort_values('Avg_Rank', ascending=True).reset_index(drop=True)
    out['Global_Rank'] = out.index + 1
    return out


def get_display_name(model):
    """Get display name for model."""
    if model in MODEL_DISPLAY:
        return MODEL_DISPLAY[model][0]
    return model

def get_category(model):
    """Get category for model."""
    if model in MODEL_DISPLAY:
        return MODEL_DISPLAY[model][1]
    return 'Other'

def get_fine_category(model):
    """Get fine-grained category for model."""
    if model in MODEL_CATEGORY:
        return MODEL_CATEGORY[model]
    return 'Other'

def get_venue(model):
    """Get publication venue for model."""
    if model in MODEL_VENUE:
        return MODEL_VENUE[model]
    return '--'
def format_metric(value_str):
    """Format metric string for LaTeX."""
    if pd.isna(value_str) or value_str == '0' or value_str == 'N/A':
        return '--'
    # Already formatted as "0.215 (0.003)" -> "0.215$\\pm$0.003"
    if '(' in str(value_str):
        parts = str(value_str).replace(')', '').split('(')
        mean = parts[0].strip()
        std = parts[1].strip() if len(parts) > 1 else ''
        if std:
            return f"{mean}$\\pm${std}"
        return mean
    return str(value_str)

def format_size(size_str):
    """Format parameter count."""
    if pd.isna(size_str) or size_str == 'N/A' or size_str == '0':
        return 'N/A'
    try:
        # Remove commas and convert to float
        size = int(str(size_str).replace(',', '').replace('"', ''))
        if size >= 1e6:
            return f"{size/1e6:.1f}M"
        elif size >= 1e3:
            return f"{size/1e3:.1f}K"
        return str(size)
    except:
        return str(size_str)

def format_time(time_str):
    """Format inference time."""
    if pd.isna(time_str) or time_str == 'N/A' or time_str == '0':
        return 'N/A'
    try:
        time_val = float(time_str)
        return f"{time_val:.2f}s"
    except:
        return str(time_str)

def extract_numeric_value(value_str):
    if pd.isna(value_str):
        return float('inf')
    s = str(value_str).strip()

    # 统一缺失符号
    if s in {'--', 'N/A', ''}:
        return float('inf')

    # 显式处理 nan/inf 字符串
    low = s.lower()
    if low in {'nan', 'inf', '+inf', '-inf'}:
        return float('inf')

    # 去掉 ±/pm/括号等
    if '$' in s:
        s = s.split('$')[0]
    if '±' in s:
        s = s.split('±')[0]
    if '(' in s:
        s = s.split('(')[0]
    s = s.strip().replace('s', '')

    try:
        # 处理 K/M
        if s.endswith('M'):
            v = float(s[:-1]) * 1e6
        elif s.endswith('K'):
            v = float(s[:-1]) * 1e3
        else:
            v = float(s)

        # 关键：把 0 当成缺失（避免失败写 0 的情况）
        if v == 0.0:
            return float('inf')

        return v
    except:
        return float('inf')

def get_mae_from_csv(base_path, pattern, dataset, model):
    """
    Read MAE (formatted for LaTeX) for a given model from:
      base_path/pattern/{dataset}_with_naive.csv
    Return '--' if not found.
    """
    csv_path = os.path.join(base_path, pattern, f'{dataset}_with_naive.csv')
    if not os.path.exists(csv_path):
        return '--'
    df = pd.read_csv(csv_path)
    row = df[df['Model'] == model]
    if len(row) == 0:
        return '--'
    return format_metric(row.iloc[0].get('MAE', '--'))


def get_column_ranks(values):
    """
    Get ranks for a list of values (lower is better).
    Returns a dict: {index: rank} where rank is 1-based.
    """
    # Extract numeric values with indices
    indexed_values = [(i, extract_numeric_value(v)) for i, v in enumerate(values)]
    
    # Filter out inf values (invalid/missing)
    valid_values = [(i, v) for i, v in indexed_values if v != float('inf')]
    
    # Sort by value (ascending = lower is better)
    sorted_values = sorted(valid_values, key=lambda x: x[1])
    
    # Assign ranks
    ranks = {}
    for rank, (idx, val) in enumerate(sorted_values, 1):
        ranks[idx] = rank
    
    return ranks

def format_by_rank(value_str, rank):
    """
    Format value string based on rank (ICML Style).
    1st: bold
    2nd: underline
    """
    if rank == 1:
        return f"\\textbf{{{value_str}}}"       # 第一名：加粗
    elif rank == 2:
        return f"\\underline{{{value_str}}}"    # 第二名：下划线
    else:
        return value_str

# =============================================================================
# Table 1: Overall Ranking (unchanged)
# =============================================================================

def generate_table1_overall_ranking(base_path, output_dir):
    """Generate Table 1: Overall Ranking over main models only (exclude Ablation + Naive)."""

    df = compute_main_global_ranking_from_setting_csvs(base_path)

    latex = []
    latex.append(r"\begin{table*}[t]")
    latex.append(r"    \caption{Overall ranking across all experimental settings, computed among HELIX and competitive baselines (excluding ablations and most naive methods), while additionally reporting two strong naive references (LOCF and Linear Interpolation). Lower average rank indicates better performance. $\dagger$ indicates models that could not run on all settings due to computational or architectural constraints.}")
    latex.append(r"    \label{tab:main_ranking}")
    latex.append(r"    \centering")
    latex.append(r"    \begin{small}")
    latex.append(r"    \begin{tabular}{l|ccc|ll}")
    latex.append(r"        \toprule")
    latex.append(r"        \textbf{Model} & \textbf{Avg. Rank} $\downarrow$ & \textbf{Valid Exps.} & \textbf{Global Rank} & \textbf{Category} & \textbf{Venue} \\")
    latex.append(r"        \midrule")

    for _, row in df.iterrows():
        model = row['Model']
        display_name = get_display_name(model)
        category = get_fine_category(model)
        venue = get_venue(model)
        avg_rank = row['Avg_Rank']
        valid_exps = int(row['Valid_Experiments'])
        total_exps = int(row['Total_Experiments'])
        global_rank = int(row['Global_Rank'])

        # Format model name
        if model == 'HELIX':
            model_str = f"\\textbf{{{display_name}}}"
            rank_str = f"\\textbf{{{avg_rank:.2f}}}"
            global_str = f"\\textbf{{{global_rank}}}"
        else:
            model_str = display_name
            rank_str = f"{avg_rank:.2f}"
            global_str = str(global_rank)

        # Add dagger for incomplete experiments
        if valid_exps < total_exps:
            model_str += "$^\\dagger$"

        latex.append(f"        {model_str} & {rank_str} & {valid_exps}/{total_exps} & {global_str} & {category} & {venue} \\\\")

    latex.append(r"        \bottomrule")
    latex.append(r"    \end{tabular}")
    latex.append(r"    \end{small}")
    latex.append(r"\end{table*}")

    output_path = os.path.join(output_dir, 'table1_overall_ranking.tex')
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex))
    print(f"✓ Generated: {output_path}")


# =============================================================================
# Table 2: ETT-h1 Detailed Results (5 patterns + #Params + Time)
# =============================================================================

def generate_table2_detailed_results(base_path, output_dir):
    """Generate Table 2: ETT-h1 with all 5 missing patterns, #Params, Time. Top-5 ranking highlighted."""
    
    patterns = ['point01', 'point05', 'point09', 'block05', 'subseq05']
    
    # Load data for all patterns
    all_data = {}
    for pattern in patterns:
        csv_path = os.path.join(base_path, pattern, 'ETT_h1_with_naive.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                model = row['Model']
                if model not in all_data:
                    all_data[model] = {'Size': row.get('Size', 'N/A'), 'Time': row.get('Time', 'N/A')}
                all_data[model][pattern] = format_metric(row['MAE'])
    
    # Define model order for this table
    display_order = [
        # Naive first
        'Naive_Mean', 'Naive_Median', 'Naive_LOCF', 'Naive_LinearInterp',
        # Then DL methods
        'HELIX', 
        # 'HELIX_NoFusion', 'HELIX_NoRotaryPE', 'HELIX_NoHybrid', 'HELIX_NoFeatureEmbed',
        'ImputeFormer', 'SAITS',
        'NonstationaryTransformer', 'PatchTST', 'iTransformer',
        'TEFN', 'TimeMixer', 'TimeMixerPP', 'ModernTCN', 'StemGNN', 'TOTEM',
        'FreTS', 'TimeLLM', 'MOMENT',
    ]
    
    # Filter to models that exist
    display_order = [m for m in display_order if m in all_data]
    
    # Prepare column data for ranking
    columns_data = {}
    for pattern in patterns:
        columns_data[pattern] = [all_data[m].get(pattern, '--') for m in display_order]
    columns_data['Size'] = [format_size(all_data[m].get('Size', 'N/A')) for m in display_order]
    columns_data['Time'] = [format_time(all_data[m].get('Time', 'N/A')) for m in display_order]
    
    # Get ranks for each column
    column_ranks = {}
    for col_name, values in columns_data.items():
        column_ranks[col_name] = get_column_ranks(values)
    
    latex = []
    latex.append(r"\begin{table*}[t]")
    latex.append(r"    \caption{Detailed MAE results on ETT-h1 (48 steps, 7 features) across all missing patterns. Mean $\pm$ std over 5 runs. Ranking: \textbf{1st}, \underline{2nd}.}")
    latex.append(r"    \label{tab:detailed_results}")
    latex.append(r"    \centering")
    latex.append(r"    \begin{small}")
    latex.append(r"    \begin{tabular}{l|ccccc|cc}")
    latex.append(r"        \toprule")
    latex.append(r"        \textbf{Method} & \textbf{Point-10\%} & \textbf{Point-50\%} & \textbf{Point-90\%} & \textbf{Block-50\%} & \textbf{Subseq-50\%} & \textbf{\#Params}$\downarrow$ & \textbf{Time}$\downarrow$ \\")
    latex.append(r"        \midrule")
    
    # Add separator before DL methods
    naive_done = False
    
    for idx, model in enumerate(display_order):
        if model == 'HELIX' and not naive_done:
            latex.append(r"        \midrule")
            naive_done = True
        
        row_parts = [get_display_name(model)]
        
        # Add pattern columns with ranking format
        for pattern in patterns:
            val = columns_data[pattern][idx]
            rank = column_ranks[pattern].get(idx, 999)
            formatted_val = format_by_rank(val, rank) if rank <= 2 else val
            row_parts.append(formatted_val)
        
        # Add Size column (no ranking)
        size_val = columns_data['Size'][idx]
        row_parts.append(size_val)

        # Add Time column (no ranking)
        time_val = columns_data['Time'][idx]
        row_parts.append(time_val)
        
        latex.append("        " + " & ".join(row_parts) + " \\\\")
    
    latex.append(r"        \bottomrule")
    latex.append(r"    \end{tabular}")
    latex.append(r"    \end{small}")
    latex.append(r"\end{table*}")
    
    output_path = os.path.join(output_dir, 'table2_detailed_results.tex')
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex))
    print(f"✓ Generated: {output_path}")

# =============================================================================
# Table 3: Ablation Study (Transposed)
# =============================================================================

def generate_table3_ablation(base_path, output_dir):
    """Generate Table 3: Ablation study (transposed - models as columns, metrics as rows)."""
    
    # Load by-pattern data
    pattern_path = os.path.join(base_path, 'analysis', 'analysis_by_pattern.csv')
    df_pattern = pd.read_csv(pattern_path)
    
    # Ablation models in order
    ablation_models = ['HELIX', 'HELIX_NoFusion', 'HELIX_NoRotaryPE', 'HELIX_NoHybrid', 'HELIX_NoFeatureEmbed']
    ablation_display = ['HELIX', 'w/o Fusion', 'w/o Rotary', 'w/o Hybrid', 'w/o FeatEmb']
    
    # Get HELIX baseline
    helix_row = df_pattern[df_pattern['Model'] == 'HELIX'].iloc[0]
    helix_avg = helix_row['Avg_Across_Patterns']
    
    # Collect data
    data = {}
    for model in ablation_models:
        row = df_pattern[df_pattern['Model'] == model]
        if len(row) > 0:
            row = row.iloc[0]
            avg_rank = row['Avg_Across_Patterns']
            data[model] = {
                'Avg': f"{avg_rank:.2f}",
                'Delta': '--' if model == 'HELIX' else f"+{avg_rank - helix_avg:.2f}" if avg_rank > helix_avg else f"{avg_rank - helix_avg:.2f}",
                'Std': f"{row['Std_Across_Patterns']:.2f}",
                'Point-10%': f"{row['point01']:.2f}",
                'Point-50%': f"{row['point05']:.2f}",
                'Point-90%': f"{row['point09']:.2f}",
                'Block': f"{row['block05']:.2f}",
                'Subseq': f"{row['subseq05']:.2f}",
            }
    
    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"    \caption{Ablation study: Impact of each component. Avg = average rank across all patterns; $\Delta$ = change vs.\ full HELIX; Std = standard deviation across patterns (lower = more stable). Removing Feature Identity Embedding causes the largest degradation ($\Delta$=+4.02). Multi-level Fusion reduces cross-pattern variance (Std: 1.14$\to$0.31).}")
    latex.append(r"    \label{tab:ablation}")
    latex.append(r"    \centering")
    latex.append(r"    \begin{small}")
    latex.append(r"    \begin{tabular}{l|ccccc}")
    latex.append(r"        \toprule")
    
    # Header row with model names
    header = "        \\textbf{Metric} & " + " & ".join([f"\\textbf{{{name}}}" for name in ablation_display]) + " \\\\"
    latex.append(header)
    latex.append(r"        \midrule")
    
    # Row order
    row_order = ['Avg', 'Delta', 'Std', 'Point-10%', 'Point-50%', 'Point-90%', 'Block', 'Subseq']
    row_labels = {
        'Avg': 'Avg Rank $\\downarrow$',
        'Delta': '$\\Delta$',
        'Std': 'Std $\\downarrow$',
        'Point-10%': 'Point-10\\%',
        'Point-50%': 'Point-50\\%',
        'Point-90%': 'Point-90\\%',
        'Block': 'Block-50\\%',
        'Subseq': 'Subseq-50\\%',
    }
    
    for row_key in row_order:
        row_parts = [row_labels[row_key]]
        for model in ablation_models:
            val = data[model][row_key]
            # Bold HELIX column values for Avg and Std
            if model == 'HELIX' and row_key in ['Avg', 'Std']:
                val = f"\\textbf{{{val}}}"
            row_parts.append(val)
        
        latex.append("        " + " & ".join(row_parts) + " \\\\")
        
        # Add midrule after Std
        if row_key == 'Std':
            latex.append(r"        \midrule")
    
    latex.append(r"        \bottomrule")
    latex.append(r"    \end{tabular}")
    latex.append(r"    \end{small}")
    latex.append(r"\end{table}")
    
    output_path = os.path.join(output_dir, 'table3_ablation.tex')
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex))
    print(f"✓ Generated: {output_path}")

# =============================================================================
# Appendix Tables: Full Results per Dataset (with Top-5 ranking + #Params + Time)
# =============================================================================

def generate_appendix_tables(base_path, output_dir):
    """Generate appendix tables: one per dataset with all missing patterns + #Params + Time, top-5 highlighted."""
    
    datasets_patterns = {
        'BeijingAir': ['point01', 'point05', 'point09', 'block05', 'subseq05'],
        'ETT_h1': ['point01', 'point05', 'point09', 'block05', 'subseq05'],
        'ItalyAir': ['point01', 'point05', 'point09', 'block05', 'subseq05'],
        'PeMS': ['point01', 'point05', 'point09', 'block05', 'subseq05'],
        'PhysioNet2012': ['point01'],  # Only point01
    }
    
    for dataset, patterns in datasets_patterns.items():
        latex = []
        
        # Determine column count (patterns + Size + Time)
        n_patterns = len(patterns)
        col_spec = 'l|' + 'c' * n_patterns + '|cc'
        
        latex.append(r"\begin{table}[h]")
        dataset_escaped = dataset.replace('_', '\\_')
        latex.append(fr"    \caption{{Complete MAE results on {dataset_escaped} across all missing patterns. Mean $\pm$ std over 5 runs. Ranking: \textbf{{1st}}, \underline{{2nd}}.}}")
        latex.append(f"    \\label{{tab:full_{dataset.lower()}}}")
        latex.append(r"    \centering")
        latex.append(r"    \begin{footnotesize}")
        latex.append(f"    \\begin{{tabular}}{{{col_spec}}}")
        latex.append(r"        \toprule")
        
        # Header
        header_parts = ['\\textbf{Model}']
        for p in patterns:
            header_parts.append(f"\\textbf{{{PATTERN_NAMES.get(p, p)}}}")
        header_parts.append('\\textbf{\\#Params}$\\downarrow$')
        header_parts.append('\\textbf{Time}$\\downarrow$')
        latex.append("        " + " & ".join(header_parts) + " \\\\")
        latex.append(r"        \midrule")
        
        # Collect data for all patterns + Size + Time
        all_data = {}
        for pattern in patterns:
            csv_path = os.path.join(base_path, pattern, f'{dataset}_with_naive.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                for _, row in df.iterrows():
                    model = row['Model']
                    if get_category(model) == 'Ablation':
                        continue
                    if model not in all_data:
                        all_data[model] = {
                            'Size': row.get('Size', 'N/A'),
                            'Time': row.get('Time', 'N/A')
                        }
                    all_data[model][pattern] = format_metric(row['MAE'])
        
        # Define display order (A-method Appendix方案1: completely hide ablations)
        display_order = [
            m for m in MODEL_ORDER
            if (m in all_data) and (get_category(m) != 'Ablation')
        ]

        
        # Prepare column data for ranking
        columns_data = {}
        for pattern in patterns:
            columns_data[pattern] = [all_data[m].get(pattern, '--') for m in display_order]
        columns_data['Size'] = [format_size(all_data[m].get('Size', 'N/A')) for m in display_order]
        columns_data['Time'] = [format_time(all_data[m].get('Time', 'N/A')) for m in display_order]
        
        # Get ranks for each column
        column_ranks = {}
        for col_name, values in columns_data.items():
            column_ranks[col_name] = get_column_ranks(values)
        
        # Group: Naive methods first
        latex.append(r"        \multicolumn{" + str(n_patterns + 3) + r"}{l}{\textit{Naive Baselines}} \\")
        naive_models = [m for m in display_order if m.startswith('Naive_')]
        for model in naive_models:
            idx = display_order.index(model)
            row_parts = [get_display_name(model)]
            
            # Add pattern columns with ranking
            for p in patterns:
                val = columns_data[p][idx]
                rank = column_ranks[p].get(idx, 999)
                formatted_val = format_by_rank(val, rank) if rank <= 2 else val
                row_parts.append(formatted_val)
            
            # Add Size column (no ranking)
            size_val = columns_data['Size'][idx]
            row_parts.append(size_val)

            # Add Time column (no ranking)
            time_val = columns_data['Time'][idx]
            row_parts.append(time_val)
            
            latex.append("        " + " & ".join(row_parts) + " \\\\")
        
        latex.append(r"        \midrule")
        latex.append(r"        \multicolumn{" + str(n_patterns + 3) + r"}{l}{\textit{Deep Learning Methods}} \\")
        
        # DL methods
        dl_models = [m for m in display_order if not m.startswith('Naive_')]
        for model in dl_models:
            idx = display_order.index(model)
            row_parts = [get_display_name(model)]
            
            # Add pattern columns with ranking
            for p in patterns:
                val = columns_data[p][idx]
                rank = column_ranks[p].get(idx, 999)
                formatted_val = format_by_rank(val, rank) if rank <= 2 else val
                row_parts.append(formatted_val)
            
            # Add Size column (no ranking)
            size_val = columns_data['Size'][idx]
            row_parts.append(size_val)

            # Add Time column (no ranking)
            time_val = columns_data['Time'][idx]
            row_parts.append(time_val)
            
            # Bold HELIX model name
            if model == 'HELIX':
                row_parts[0] = f"\\textbf{{{row_parts[0]}}}"
            
            latex.append("        " + " & ".join(row_parts) + " \\\\")
        
        latex.append(r"        \bottomrule")
        latex.append(r"    \end{tabular}")
        latex.append(r"    \end{footnotesize}")
        latex.append(r"\end{table}")
        
        output_path = os.path.join(output_dir, f'appendix_{dataset.lower()}.tex')
        with open(output_path, 'w') as f:
            f.write('\n'.join(latex))
        print(f"✓ Generated: {output_path}")

def generate_ablation_appendix_all_datasets(base_path, output_dir):
    """
    Appendix Ablation Tables (grouped by dataset, stacked vertically):
      - One table per dataset
      - Models are COLUMNS (horizontal)
      - Patterns are ROWS
      - Rank per row among ablations only: 1st bold, 2nd underline
    Output: appendix_ablation_by_dataset.tex
    """

    datasets_patterns = {
        'BeijingAir': ['point01', 'point05', 'point09', 'block05', 'subseq05'],
        'ETT_h1': ['point01', 'point05', 'point09', 'block05', 'subseq05'],
        'ItalyAir': ['point01', 'point05', 'point09', 'block05', 'subseq05'],
        'PeMS': ['point01', 'point05', 'point09', 'block05', 'subseq05'],
        'PhysioNet2012': ['point01'],
    }

    ablation_models = [
        'HELIX',
        'HELIX_NoFusion',
        'HELIX_NoRotaryPE',
        'HELIX_NoHybrid',
        'HELIX_NoFeatureEmbed'
    ]
    ablation_display = {
        'HELIX': 'HELIX (Ours)',
        'HELIX_NoFusion': 'w/o Multi-level Fusion',
        'HELIX_NoRotaryPE': 'w/o Rotary PE',
        'HELIX_NoHybrid': 'w/o Hybrid Encoding',
        'HELIX_NoFeatureEmbed': 'w/o Feature Identity Emb.',
    }

    out = []
    out.append("% Auto-generated: Ablation appendix tables grouped by dataset (horizontal).")
    out.append("% Each dataset is a separate table; tables are stacked vertically in this file.")
    out.append("")

    for dataset, patterns in datasets_patterns.items():
        # Build matrix: rows=patterns, cols=models
        row_values = {}  # p -> [val_model1, ...]
        for p in patterns:
            row_values[p] = [get_mae_from_csv(base_path, p, dataset, m) for m in ablation_models]

        # Rank per row (pattern): among ablation models only
        row_ranks = {p: get_column_ranks(row_values[p]) for p in patterns}

        dataset_title = DATASET_NAMES.get(dataset, dataset).replace('_', '\\_')

        out.append(r"\begin{table*}[t]")
        out.append(r"    \centering")
        out.append(fr"    \caption{{Ablation results (MAE) on {dataset_title}. Ranking per row among ablations: \textbf{{1st}}, \underline{{2nd}}.}}")
        out.append(fr"    \label{{tab:ablation_{dataset.lower()}_appendix}}")
        out.append(r"    \begin{footnotesize}")
        out.append(r"    \resizebox{\textwidth}{!}{%")

        # 1st col for pattern name + 5 model cols
        col_spec = "l|" + "c" * len(ablation_models)
        out.append(f"    \\begin{{tabular}}{{{col_spec}}}")
        out.append(r"        \toprule")

        # Header
        header = [r"\textbf{Pattern}"]
        for m in ablation_models:
            name = ablation_display.get(m, get_display_name(m))
            if m == 'HELIX':
                name = r"\textbf{" + name + r"}"
            header.append(r"\textbf{" + name + r"}")
        out.append("        " + " & ".join(header) + r" \\")
        out.append(r"        \midrule")

        # Body: each pattern is a row
        for p in patterns:
            row = [r"\textbf{" + PATTERN_NAMES.get(p, p) + r"}"]
            ranks = row_ranks[p]  # {idx: rank}
            for j, m in enumerate(ablation_models):
                val = row_values[p][j]
                rank = ranks.get(j, 999)
                val = format_by_rank(val, rank) if rank <= 2 else val
                row.append(val)
            out.append("        " + " & ".join(row) + r" \\")

        out.append(r"        \bottomrule")
        out.append(r"    \end{tabular}}")  # end resizebox
        out.append(r"    \end{footnotesize}")
        out.append(r"\end{table*}")
        out.append("")  # blank line between dataset blocks

    output_path = os.path.join(output_dir, 'appendix_ablation_by_dataset.tex')
    with open(output_path, 'w') as f:
        f.write('\n'.join(out))
    print(f"✓ Generated: {output_path}")


def generate_ablation_beijingair_main(base_path, output_dir):
    """
    Main paper small ablation table on BeijingAir only (5 patterns).
    Rows: HELIX + 4 ablations
    Cols: 5 missing patterns
    """
    dataset = 'BeijingAir'
    patterns = [ 'point05', 'block05', 'subseq05']#'point01', 'point09',

    ablation_models = ['HELIX', 'HELIX_NoFusion', 'HELIX_NoRotaryPE', 'HELIX_NoHybrid', 'HELIX_NoFeatureEmbed']
    ablation_display = {
        'HELIX': 'HELIX (Ours)',
        'HELIX_NoFusion': 'w/o Fusion',
        'HELIX_NoRotaryPE': 'w/o Rotary',
        'HELIX_NoHybrid': 'w/o Hybrid',
        'HELIX_NoFeatureEmbed': 'w/o FeatEmb',
    }

    # Collect values
    all_data = {m: {} for m in ablation_models}
    for p in patterns:
        for m in ablation_models:
            all_data[m][p] = get_mae_from_csv(base_path, p, dataset, m)

    # Column-wise ranks among ablations only
    columns_data = {p: [all_data[m][p] for m in ablation_models] for p in patterns}
    column_ranks = {p: get_column_ranks(columns_data[p]) for p in patterns}

    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"    \caption{Ablation on BeijingAir across missing patterns (MAE). Ranking per column among ablations: \textbf{1st}, \underline{2nd}.}")
    latex.append(r"    \label{tab:ablation_beijingair}")
    latex.append(r"    \centering")
    latex.append(r"    \begin{small}")
    latex.append(r"    \begin{tabular}{l|ccc}")
    latex.append(r"        \toprule")
    latex.append(r"        \textbf{Model} & \textbf{Point-50\%} & \textbf{Block-50\%} & \textbf{Subseq-50\%} \\")
    latex.append(r"        \midrule")

    for i, m in enumerate(ablation_models):
        name = ablation_display.get(m, get_display_name(m))
        if m == 'HELIX':
            name = r"\textbf{" + name + r"}"

        row = [name]
        for p in patterns:
            val = columns_data[p][i]
            rank = column_ranks[p].get(i, 999)
            val = format_by_rank(val, rank) if rank <= 2 else val
            row.append(val)

        latex.append("        " + " & ".join(row) + r" \\")

    latex.append(r"        \bottomrule")
    latex.append(r"    \end{tabular}")
    latex.append(r"    \end{small}")
    latex.append(r"\end{table}")

    output_path = os.path.join(output_dir, 'table3_ablation_beijingair.tex')
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex))
    print(f"✓ Generated: {output_path}")

# =============================================================================
# Additional: vs Naive Summary
# =============================================================================

def generate_vs_naive_summary(base_path, output_dir):
    """Generate summary of improvement vs Linear Interpolation."""
    
    csv_path = os.path.join(base_path, 'analysis', 'analysis_vs_naive_summary.csv')
    if not os.path.exists(csv_path):
        print(f"⚠ Skipped: {csv_path} not found")
        return
    
    df = pd.read_csv(csv_path)
    
    # Sort by improvement
    if 'Avg_Improvement_vs_LinearInterp' in df.columns:
        sort_col = 'Avg_Improvement_vs_LinearInterp'
    else:
        # Try to find similar column
        improvement_cols = [c for c in df.columns if 'Improvement' in c or 'improvement' in c]
        if improvement_cols:
            sort_col = improvement_cols[0]
        else:
            print(f"⚠ Skipped: No improvement column found")
            return
    
    df = df.sort_values(sort_col, ascending=False)
    
    latex = []
    latex.append("% " + "=" * 70)
    latex.append("% VS NAIVE SUMMARY")
    latex.append("% " + "=" * 70)
    latex.append("% Average improvement over Linear Interpolation:")
    latex.append("%")
    
    for i, (_, row) in enumerate(df.head(15).iterrows()):
        model = row['Model']
        improvement = row[sort_col]
        sign = "+" if improvement > 0 else ""
        latex.append(f"% {i+1:2d}. {get_display_name(model):30s}: {sign}{improvement:.1f}%")
    
    latex.append("%")
    latex.append("% KEY INSIGHT FOR WRITING:")
    latex.append("% Many DL methods FAIL to beat Linear Interpolation:")
    latex.append("% - PatchTST, FreTS, iTransformer all negative")
    latex.append("% - HELIX achieves +37.3%, proving meaningful structure learning")
    
    output_path = os.path.join(output_dir, 'vs_naive_summary.tex')
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex))
    print(f"✓ Generated: {output_path}")

# =============================================================================
# Additional Reference Data
# =============================================================================

def generate_reference_data(base_path, output_dir):
    """Generate reference data files for writing (as comments)."""
    
    latex = []
    latex.append("% " + "=" * 70)
    latex.append("% REFERENCE DATA FOR PAPER WRITING")
    latex.append("% " + "=" * 70)
    latex.append("")
    
    # By-pattern analysis
    pattern_path = os.path.join(base_path, 'analysis', 'analysis_by_pattern.csv')
    if os.path.exists(pattern_path):
        df_pattern = pd.read_csv(pattern_path)
        latex.append("% " + "-" * 70)
        latex.append("% BY-PATTERN RANKING (Full data)")
        latex.append("% " + "-" * 70)
        latex.append("% Model, point01, point05, point09, block05, subseq05, Avg, Std")
        latex.append("%")
        for _, row in df_pattern.head(15).iterrows():
            model = row['Model']
            latex.append(f"% {get_display_name(model):30s}: pt01={row['point01']:.2f}, pt05={row['point05']:.2f}, pt09={row['point09']:.2f}, block={row['block05']:.2f}, subseq={row['subseq05']:.2f}, avg={row['Avg_Across_Patterns']:.2f}, std={row['Std_Across_Patterns']:.2f}")
        latex.append("")
    
    # Key writing points summary
    latex.append("% " + "=" * 70)
    latex.append("% SUMMARY: KEY POINTS FOR PAPER")
    latex.append("% " + "=" * 70)
    latex.append("%")
    latex.append("% 1. HELIX achieves rank 1 overall (avg rank 2.97)")
    latex.append("%")
    latex.append("% 2. Ablation insights:")
    latex.append("%    a) Feature Identity Embedding is MOST critical (Δ=+4.02)")
    latex.append("%    b) Multi-level Fusion = stability trade-off (Std: 1.14 → 0.31)")
    
    output_path = os.path.join(output_dir, 'reference_data.tex')
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex))
    print(f"✓ Generated: {output_path}")

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate LaTeX tables from CSV results')
    parser.add_argument('--base_path', type=str, 
                        default='/home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/reproduce_imputation/results_csv/imputation',
                        help='Base path to results_csv/imputation directory')
    parser.add_argument('--output_dir', type=str, default='./latex_tables',
                        help='Output directory for LaTeX files')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Generating LaTeX Tables for HELIX Paper (v2)")
    print("=" * 60)
    print(f"Input: {args.base_path}")
    print(f"Output: {args.output_dir}")
    print()
    
    # Generate all tables
    print("Generating Table 1: Overall Ranking...")
    generate_table1_overall_ranking(args.base_path, args.output_dir)
    
    print("\nGenerating Table 2: ETT-h1 Detailed Results...")
    generate_table2_detailed_results(args.base_path, args.output_dir)
    
    print("\nGenerating Ablation Table (Main): BeijingAir 5 patterns...")
    generate_ablation_beijingair_main(args.base_path, args.output_dir)
    
    print("\nGenerating Appendix Tables (6 datasets)...")
    generate_appendix_tables(args.base_path, args.output_dir)
    
    print("\nGenerating Ablation Table (Appendix): All datasets (wide)...")
    generate_ablation_appendix_all_datasets(args.base_path, args.output_dir)
    
    print("\nGenerating vs Naive Summary...")
    generate_vs_naive_summary(args.base_path, args.output_dir)
    
    print("\nGenerating Reference Data...")
    generate_reference_data(args.base_path, args.output_dir)
    
    print("\n" + "=" * 60)
    print("Done! All LaTeX tables generated.")
    print("=" * 60)

if __name__ == "__main__":
    main()