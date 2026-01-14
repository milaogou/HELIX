"""
Figure 5: Imputation Effect Visualization for HELIX Paper
ICML 2026 Submission - Best Paper Quality

This script generates a comprehensive visualization comparing HELIX against
baselines across three missing patterns (Point-50%, Block-50%, Subseq-50%).

Layout:
    Row 1: Time series comparison (3 panels)
    Row 2: Quantitative analysis (3 panels)
        - (d) Error distribution (violin plot)
        - (e) Boundary analysis (MAE at missing region edges)
        - (f) Cross-station correlation utilization

Author: Generated for HELIX ICML 2026 submission
"""

import os
import sys
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

# Add path for pickle_load
sys.path.insert(0, '/home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code')
from pypots.data.saving import pickle_load

# =============================================================================
# Configuration
# =============================================================================

# Style settings - ICML/Nature quality
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
    'axes.spines.top': True,
    'axes.spines.right': True,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
})

# Color scheme - distinct and colorblind-friendly
COLORS = {
    'ground_truth': '#2c3e50',      # Dark blue-gray
    'HELIX': '#e74c3c',             # Red (ours - prominent)
    'ImputeFormer': '#3498db',      # Blue
    'SAITS': '#27ae60',             # Green
    'missing_region': '#ecf0f1',    # Light gray for missing regions
    'missing_border': '#95a5a6',    # Gray border
}

LINE_STYLES = {
    'ground_truth': '-',
    'HELIX': '-',
    'ImputeFormer': '--',
    'SAITS': ':',
}

LINE_WIDTHS = {
    'ground_truth': 1.5,
    'HELIX': 1.8,
    'ImputeFormer': 1.2,
    'SAITS': 1.2,
}

# Beijing station info
STATION_NAMES = ['Huairou', 'Shunyi', 'Wanliu', 'Dingling', 'Nongzhanguan',
                 'Dongsi', 'Tiantan', 'Wanshouxigong', 'Aotizhongxin',
                 'Changping', 'Guanyuan', 'Gucheng']

STATION_SHORT = {
    'Huairou': 'HR', 'Shunyi': 'SY', 'Wanliu': 'WL', 'Dingling': 'DL',
    'Nongzhanguan': 'NZ', 'Dongsi': 'DS', 'Tiantan': 'TT',
    'Wanshouxigong': 'WS', 'Aotizhongxin': 'AZ', 'Changping': 'CP',
    'Guanyuan': 'GY', 'Gucheng': 'GC'
}

N_STATIONS = 12
N_FEATURES_PER_STATION = 11
N_STEPS = 24

# =============================================================================
# Path Configuration
# =============================================================================

BASE_DATA_PATH = '/home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/data/generated_datasets'
BASE_RESULT_PATH = '/home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/reproduce_imputation'

PATTERN_CONFIG = {
    'Point-50%': {
        'data_dir': 'beijing_air_quality_rate05_step24_point',
        'result_dir': 'point05_log/BeijingAir_log',
    },
    'Block-50%': {
        'data_dir': 'beijing_air_quality_rate00_step24_block_blocklen6',
        'result_dir': 'block00_log/BeijingAir_log',
    },
    'Subseq-50%': {
        'data_dir': 'beijing_air_quality_rate05_step24_subseq_seqlen18',
        'result_dir': 'subseq05_log/BeijingAir_log',
    },
}

MODEL_NAMES = ['HELIX', 'ImputeFormer', 'SAITS']
MODEL_DIRS = {
    'HELIX': 'HELIX_BeijingAir',
    'ImputeFormer': 'ImputeFormer_BeijingAir',
    'SAITS': 'SAITS_BeijingAir',
}


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_ground_truth(pattern_name, split='test'):
    """Load ground truth data (X_ori) and observed data (X) with mask."""
    config = PATTERN_CONFIG[pattern_name]
    data_path = os.path.join(BASE_DATA_PATH, config['data_dir'], f'{split}.h5')
    
    with h5py.File(data_path, 'r') as f:
        X_ori = f['X_ori'][:]  # Ground truth
        X = f['X'][:]          # Observed (with NaN for missing)
    
    # Create missing mask: 1 = observed, 0 = missing (artificially)
    # Note: X has NaN for missing values
    artificial_missing_mask = ~np.isnan(X)
    
    # Also track where X_ori has valid values (not naturally missing)
    valid_mask = ~np.isnan(X_ori)
    
    # Combined: positions that are artificially missing AND have valid ground truth
    # This is important for computing errors
    
    return X_ori, X, artificial_missing_mask, valid_mask


def load_imputation_results(pattern_name, model_name, n_rounds=5):
    """Load imputation results, averaged across rounds."""
    config = PATTERN_CONFIG[pattern_name]
    model_dir = MODEL_DIRS[model_name]
    
    imputations = []
    for round_idx in range(n_rounds):
        pkl_path = os.path.join(
            BASE_RESULT_PATH,
            config['result_dir'],
            model_dir,
            f'round_{round_idx}',
            'imputation.pkl'
        )
        
        if os.path.exists(pkl_path):
            data = pickle_load(pkl_path)
            imputations.append(data['test_set_imputation'])
        else:
            print(f"Warning: {pkl_path} not found")
    
    if imputations:
        return np.mean(np.stack(imputations), axis=0)
    return None


def select_representative_sample(X_ori, missing_mask, valid_mask, pattern_name):
    """
    Select a representative sample AND feature for visualization.
    Criteria: 
    - Has clear missing regions (contiguous for Block/Subseq)
    - Shows interesting patterns (variation, trends)
    - Missing rate close to target
    - Feature has valid ground truth values
    
    Returns: (sample_idx, feature_idx)
    """
    n_samples, T, F = X_ori.shape
    
    best_score = -1
    best_sample = 0
    best_feature = 0
    
    # Try multiple samples and features
    for sample_idx in range(min(n_samples, 100)):  # Check first 100 samples
        for feat_idx in range(0, F, N_FEATURES_PER_STATION):  # Check first feature of each station
            mask = missing_mask[sample_idx, :, feat_idx]
            valid = valid_mask[sample_idx, :, feat_idx]
            
            # Skip if no valid ground truth
            if valid.sum() < T * 0.8:
                continue
            
            # Count missing rate (only among valid positions)
            missing_rate = 1 - mask[valid].mean() if valid.sum() > 0 else 0
            
            # Prefer samples with ~50% missing
            rate_score = 1 - abs(missing_rate - 0.5)
            
            # For Block/Subseq, prefer contiguous missing regions
            regions = find_missing_regions(mask.astype(int))
            if 'Block' in pattern_name or 'Subseq' in pattern_name:
                # Prefer longer contiguous regions
                max_region_len = max([e - s for s, e in regions]) if regions else 0
                region_score = min(max_region_len / 10, 1.0)  # Normalize
            else:
                region_score = 0.5  # Neutral for Point
            
            # Prefer samples with variation
            gt = X_ori[sample_idx, :, feat_idx]
            gt_valid = gt[valid]
            if len(gt_valid) > 1:
                variation = np.std(gt_valid) / (np.mean(np.abs(gt_valid)) + 1e-8)
                variation_score = min(variation, 1.0)
            else:
                variation_score = 0
            
            # Combined score
            score = rate_score * 0.3 + region_score * 0.4 + variation_score * 0.3
            
            if score > best_score and missing_rate > 0.1:  # Ensure some missing
                best_score = score
                best_sample = sample_idx
                best_feature = feat_idx
    
    print(f"  Selected sample {best_sample}, feature {best_feature} "
          f"(station: {STATION_NAMES[best_feature // N_FEATURES_PER_STATION]})")
    
    return best_sample, best_feature


def find_missing_regions(mask_1d):
    """
    Find contiguous missing regions in a 1D mask.
    
    Args:
        mask_1d: 1D array where 0 (or False) = missing, 1 (or True) = observed
    
    Returns:
        List of (start, end) tuples for missing regions
    """
    regions = []
    in_region = False
    start = 0
    
    # Convert to int if boolean
    mask = np.asarray(mask_1d).astype(int)
    
    for i, val in enumerate(mask):
        if val == 0 and not in_region:  # Start of missing region
            in_region = True
            start = i
        elif val == 1 and in_region:  # End of missing region
            in_region = False
            regions.append((start, i))
    
    if in_region:  # Handle case where missing extends to end
        regions.append((start, len(mask)))
    
    return regions


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_time_series_panel(ax, X_ori, X_obs, missing_mask, imputations, 
                           feature_idx, pattern_name, sample_idx):
    """
    Plot a single time series comparison panel.
    
    Args:
        ax: matplotlib axis
        X_ori: ground truth [T, F]
        X_obs: observed data [T, F]
        missing_mask: mask [T, F], 1=observed, 0=missing
        imputations: dict of {model_name: imputed_data [T, F]}
        feature_idx: which feature to plot
        pattern_name: for title
        sample_idx: sample index for reference
    """
    T = X_ori.shape[0]
    t = np.arange(T)
    
    gt = X_ori[:, feature_idx]
    mask = missing_mask[:, feature_idx]
    
    # Find missing regions and shade them
    missing_regions = find_missing_regions(mask)
    for start, end in missing_regions:
        ax.axvspan(start - 0.5, end - 0.5, 
                   facecolor=COLORS['missing_region'], 
                   edgecolor=COLORS['missing_border'],
                   linewidth=0.5, alpha=0.7, zorder=0)
    
    # Plot ground truth
    ax.plot(t, gt, color=COLORS['ground_truth'], 
            linestyle=LINE_STYLES['ground_truth'],
            linewidth=LINE_WIDTHS['ground_truth'],
            label='Ground Truth', zorder=3)
    
    # Plot imputations
    for model_name in ['HELIX', 'ImputeFormer', 'SAITS']:
        if model_name in imputations and imputations[model_name] is not None:
            imp = imputations[model_name][:, feature_idx]
            ax.plot(t, imp, color=COLORS[model_name],
                   linestyle=LINE_STYLES[model_name],
                   linewidth=LINE_WIDTHS[model_name],
                   label=model_name if model_name != 'HELIX' else 'HELIX (Ours)',
                   zorder=4 if model_name == 'HELIX' else 2)
    
    # Formatting
    ax.set_xlim(-0.5, T - 0.5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Normalized Value')
    
    # Get station name from feature index
    station_idx = feature_idx // N_FEATURES_PER_STATION
    station_name = STATION_NAMES[station_idx] if station_idx < len(STATION_NAMES) else f'Station {station_idx}'
    
    ax.set_title(f'({chr(97 + list(PATTERN_CONFIG.keys()).index(pattern_name))}) {pattern_name}\n'
                 f'Station: {station_name}', fontweight='bold')
    
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.3)


def plot_error_distribution(ax, errors_by_pattern, methods):
    """
    Plot violin plot of error distributions across patterns.
    
    Args:
        ax: matplotlib axis
        errors_by_pattern: dict {pattern: {method: errors_array}}
        methods: list of method names
    """
    positions = []
    data = []
    colors = []
    labels = []
    
    pattern_names = list(errors_by_pattern.keys())
    n_patterns = len(pattern_names)
    n_methods = len(methods)
    
    width = 0.8 / n_methods
    
    for p_idx, pattern in enumerate(pattern_names):
        for m_idx, method in enumerate(methods):
            if method in errors_by_pattern[pattern]:
                pos = p_idx + (m_idx - n_methods/2 + 0.5) * width
                positions.append(pos)
                err = errors_by_pattern[pattern][method]
                # Flatten and sample if too large
                err_flat = err.flatten()
                if len(err_flat) > 10000:
                    err_flat = np.random.choice(err_flat, 10000, replace=False)
                data.append(err_flat)
                colors.append(COLORS[method])
    
    # Create violin plot
    parts = ax.violinplot(data, positions=positions, widths=width*0.9,
                          showmeans=True, showmedians=False, showextrema=False)
    
    # Color the violins
    for idx, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[idx])
        pc.set_alpha(0.7)
    
    parts['cmeans'].set_color('black')
    parts['cmeans'].set_linewidth(1)
    
    # X-axis
    ax.set_xticks(range(n_patterns))
    ax.set_xticklabels(pattern_names, fontsize=9)
    ax.set_xlabel('Missing Pattern')
    ax.set_ylabel('Absolute Error')
    ax.set_title('(d) Error Distribution by Pattern', fontweight='bold')
    
    # Legend
    legend_elements = [plt.Rectangle((0,0), 1, 1, facecolor=COLORS[m], alpha=0.7, 
                                      label=m if m != 'HELIX' else 'HELIX (Ours)') 
                       for m in methods]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=7)
    
    ax.grid(True, alpha=0.3, axis='y')


def compute_boundary_errors(X_ori, imputations, missing_mask, valid_mask, boundary_width=2):
    """
    Compute errors specifically at missing region boundaries.
    
    Args:
        X_ori: ground truth [N, T, F]
        imputations: dict {method: imputed [N, T, F]}
        missing_mask: mask [N, T, F], 1=observed, 0=missing
        valid_mask: [N, T, F], True where X_ori is valid (not NaN)
        boundary_width: how many steps from boundary to consider
    
    Returns:
        dict {method: boundary_mae}
    """
    results = {}
    
    for method, imp in imputations.items():
        if imp is None:
            continue
        
        boundary_errors = []
        
        for sample_idx in range(X_ori.shape[0]):
            for feat_idx in range(X_ori.shape[2]):
                mask = missing_mask[sample_idx, :, feat_idx].astype(int)
                valid = valid_mask[sample_idx, :, feat_idx]
                regions = find_missing_regions(mask)
                
                for start, end in regions:
                    # Left boundary
                    left_start = max(0, start)
                    left_end = min(start + boundary_width, end)
                    for t in range(left_start, left_end):
                        if valid[t]:  # Only if ground truth is valid
                            err = abs(X_ori[sample_idx, t, feat_idx] - imp[sample_idx, t, feat_idx])
                            boundary_errors.append(err)
                    
                    # Right boundary
                    right_start = max(start, end - boundary_width)
                    right_end = end
                    for t in range(right_start, right_end):
                        if t > left_end - 1 and valid[t]:  # Avoid double counting
                            err = abs(X_ori[sample_idx, t, feat_idx] - imp[sample_idx, t, feat_idx])
                            boundary_errors.append(err)
        
        if boundary_errors:
            results[method] = np.mean(boundary_errors)
    
    return results


def plot_boundary_analysis(ax, boundary_errors_by_pattern, methods):
    """
    Plot bar chart of boundary errors across patterns.
    """
    pattern_names = list(boundary_errors_by_pattern.keys())
    n_patterns = len(pattern_names)
    n_methods = len(methods)
    
    x = np.arange(n_patterns)
    width = 0.8 / n_methods
    
    for m_idx, method in enumerate(methods):
        values = []
        for pattern in pattern_names:
            if method in boundary_errors_by_pattern[pattern]:
                values.append(boundary_errors_by_pattern[pattern][method])
            else:
                values.append(0)
        
        offset = (m_idx - n_methods/2 + 0.5) * width
        label = method if method != 'HELIX' else 'HELIX (Ours)'
        ax.bar(x + offset, values, width * 0.9, 
               color=COLORS[method], label=label, alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(pattern_names, fontsize=9)
    ax.set_xlabel('Missing Pattern')
    ax.set_ylabel('MAE at Boundaries')
    ax.set_title('(e) Boundary Region Analysis', fontweight='bold')
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')


def compute_station_correlation_benefit(X_ori, imputations, missing_mask, valid_mask):
    """
    Analyze how well each method utilizes cross-station correlations.
    
    For each missing value, compute:
    1. The correlation of that station with other observed stations
    2. The imputation error
    
    HELIX should show lower error when correlation is high.
    
    Returns:
        dict {method: (correlations, errors)}
    """
    N, T, F = X_ori.shape
    n_stations = N_STATIONS
    n_feats = N_FEATURES_PER_STATION
    
    results = {}
    
    for method, imp in imputations.items():
        if imp is None:
            continue
        
        correlations = []
        errors = []
        
        for sample_idx in range(min(N, 200)):  # Limit for speed
            for t in range(T):
                for s in range(n_stations):
                    # Get first feature of this station
                    feat_idx = s * n_feats
                    
                    # Check if this position is: missing AND has valid ground truth
                    if (missing_mask[sample_idx, t, feat_idx] == 0 and 
                        valid_mask[sample_idx, t, feat_idx]):
                        
                        # Compute correlation with other observed stations at this time
                        corrs = []
                        for other_s in range(n_stations):
                            if other_s != s:
                                other_feat = other_s * n_feats
                                # Other station should be observed
                                if missing_mask[sample_idx, t, other_feat] == 1:
                                    # Get valid time series for correlation
                                    gt1 = X_ori[sample_idx, :, feat_idx]
                                    gt2 = X_ori[sample_idx, :, other_feat]
                                    valid_both = valid_mask[sample_idx, :, feat_idx] & valid_mask[sample_idx, :, other_feat]
                                    
                                    if valid_both.sum() > 5:  # Need enough points
                                        c = np.corrcoef(gt1[valid_both], gt2[valid_both])[0, 1]
                                        if not np.isnan(c):
                                            corrs.append(abs(c))
                        
                        if corrs:
                            avg_corr = np.mean(corrs)
                            err = abs(X_ori[sample_idx, t, feat_idx] - imp[sample_idx, t, feat_idx])
                            correlations.append(avg_corr)
                            errors.append(err)
        
        if correlations:
            results[method] = (np.array(correlations), np.array(errors))
            print(f"    {method}: {len(correlations)} data points for correlation analysis")
    
    return results


def plot_correlation_analysis(ax, corr_results, methods):
    """
    Plot scatter/binned analysis of error vs correlation.
    """
    # Bin the correlations and compute mean error per bin
    n_bins = 5
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    for method in methods:
        if method not in corr_results:
            continue
        
        corrs, errs = corr_results[method]
        
        bin_means = []
        bin_stds = []
        for i in range(n_bins):
            mask = (corrs >= bin_edges[i]) & (corrs < bin_edges[i+1])
            if mask.sum() > 0:
                bin_means.append(np.mean(errs[mask]))
                bin_stds.append(np.std(errs[mask]) / np.sqrt(mask.sum()))
            else:
                bin_means.append(np.nan)
                bin_stds.append(np.nan)
        
        label = method if method != 'HELIX' else 'HELIX (Ours)'
        ax.errorbar(bin_centers, bin_means, yerr=bin_stds,
                   color=COLORS[method], marker='o', markersize=6,
                   linewidth=1.5, capsize=3, label=label,
                   linestyle=LINE_STYLES[method])
    
    ax.set_xlabel('Avg. Correlation with Observed Stations')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('(f) Cross-Station Correlation Utilization', fontweight='bold')
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)


# =============================================================================
# Main Figure Generation
# =============================================================================

def generate_figure5(output_dir):
    """Generate the complete Figure 5."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("Generating Figure 5: Imputation Visualization")
    print("=" * 70)
    
    # Storage for all data
    all_X_ori = {}
    all_missing_mask = {}
    all_valid_mask = {}
    all_imputations = {}
    selected_samples = {}
    selected_features = {}
    
    # Load data for each pattern
    for pattern_name in PATTERN_CONFIG.keys():
        print(f"\nLoading data for {pattern_name}...")
        
        # Load ground truth (now returns 4 values)
        X_ori, X_obs, missing_mask, valid_mask = load_ground_truth(pattern_name)
        all_X_ori[pattern_name] = X_ori
        all_missing_mask[pattern_name] = missing_mask
        all_valid_mask[pattern_name] = valid_mask
        print(f"  Shape: {X_ori.shape}")
        print(f"  Valid ratio in X_ori: {valid_mask.mean():.2%}")
        
        # Load imputations
        imputations = {}
        for model_name in MODEL_NAMES:
            print(f"  Loading {model_name}...")
            imp = load_imputation_results(pattern_name, model_name)
            if imp is not None:
                imputations[model_name] = imp
                print(f"    Loaded: shape {imp.shape}")
            else:
                print(f"    Not found!")
        all_imputations[pattern_name] = imputations
        
        # Select representative sample AND feature (now returns both)
        sample_idx, feature_idx = select_representative_sample(
            X_ori, missing_mask, valid_mask, pattern_name
        )
        selected_samples[pattern_name] = sample_idx
        selected_features[pattern_name] = feature_idx
    
    # Create figure
    print("\n" + "-" * 70)
    print("Creating figure...")
    
    fig = plt.figure(figsize=(15, 9))
    
    # Create grid: 2 rows, 3 columns
    gs = gridspec.GridSpec(2, 3, figure=fig, 
                           height_ratios=[1, 1],
                           hspace=0.35, wspace=0.25,
                           left=0.06, right=0.98, top=0.93, bottom=0.08)
    
    # Row 1: Time series panels
    print("  Plotting time series panels...")
    for idx, pattern_name in enumerate(PATTERN_CONFIG.keys()):
        ax = fig.add_subplot(gs[0, idx])
        
        sample_idx = selected_samples[pattern_name]
        feature_idx = selected_features[pattern_name]
        
        X_ori = all_X_ori[pattern_name][sample_idx]
        missing_mask = all_missing_mask[pattern_name][sample_idx]
        
        # Get imputations for this sample
        sample_imputations = {}
        for model_name, full_imp in all_imputations[pattern_name].items():
            if full_imp is not None:
                sample_imputations[model_name] = full_imp[sample_idx]
        
        plot_time_series_panel(ax, X_ori, None, missing_mask, 
                              sample_imputations, feature_idx, pattern_name, sample_idx)
    
    # Add shared legend for Row 1
    handles = [
        Line2D([0], [0], color=COLORS['ground_truth'], linestyle=LINE_STYLES['ground_truth'],
               linewidth=LINE_WIDTHS['ground_truth'], label='Ground Truth'),
        Line2D([0], [0], color=COLORS['HELIX'], linestyle=LINE_STYLES['HELIX'],
               linewidth=LINE_WIDTHS['HELIX'], label='HELIX (Ours)'),
        Line2D([0], [0], color=COLORS['ImputeFormer'], linestyle=LINE_STYLES['ImputeFormer'],
               linewidth=LINE_WIDTHS['ImputeFormer'], label='ImputeFormer'),
        Line2D([0], [0], color=COLORS['SAITS'], linestyle=LINE_STYLES['SAITS'],
               linewidth=LINE_WIDTHS['SAITS'], label='SAITS'),
        Rectangle((0, 0), 1, 1, facecolor=COLORS['missing_region'], 
                  edgecolor=COLORS['missing_border'], label='Missing Region'),
    ]
    fig.legend(handles=handles, loc='upper center', ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, 0.99), frameon=True, framealpha=0.9)
    
    # Row 2: Quantitative analysis
    print("  Computing quantitative metrics...")
    
    # Compute errors for violin plot (only where both missing AND valid ground truth)
    errors_by_pattern = {}
    for pattern_name in PATTERN_CONFIG.keys():
        errors_by_pattern[pattern_name] = {}
        X_ori = all_X_ori[pattern_name]
        mask = all_missing_mask[pattern_name]
        valid = all_valid_mask[pattern_name]
        
        for model_name, imp in all_imputations[pattern_name].items():
            if imp is not None:
                # Only compute error on: artificially missing AND valid ground truth
                eval_positions = (mask == 0) & valid
                if eval_positions.sum() > 0:
                    errors = np.abs(X_ori[eval_positions] - imp[eval_positions])
                    errors_by_pattern[pattern_name][model_name] = errors
                    print(f"    {pattern_name} {model_name}: {eval_positions.sum()} positions, MAE={np.mean(errors):.4f}")
    
    # (d) Error distribution
    print("  Plotting error distribution...")
    ax_d = fig.add_subplot(gs[1, 0])
    plot_error_distribution(ax_d, errors_by_pattern, MODEL_NAMES)
    
    # Compute boundary errors
    print("  Computing boundary errors...")
    boundary_errors_by_pattern = {}
    for pattern_name in PATTERN_CONFIG.keys():
        boundary_errors_by_pattern[pattern_name] = compute_boundary_errors(
            all_X_ori[pattern_name],
            all_imputations[pattern_name],
            all_missing_mask[pattern_name],
            all_valid_mask[pattern_name]
        )
    
    # (e) Boundary analysis
    print("  Plotting boundary analysis...")
    ax_e = fig.add_subplot(gs[1, 1])
    plot_boundary_analysis(ax_e, boundary_errors_by_pattern, MODEL_NAMES)
    
    # Compute correlation benefit (using Point-50% as it's clearest)
    print("  Computing correlation analysis...")
    corr_results = compute_station_correlation_benefit(
        all_X_ori['Point-50%'],
        all_imputations['Point-50%'],
        all_missing_mask['Point-50%'],
        all_valid_mask['Point-50%']
    )
    
    # (f) Correlation analysis
    print("  Plotting correlation analysis...")
    ax_f = fig.add_subplot(gs[1, 2])
    plot_correlation_analysis(ax_f, corr_results, MODEL_NAMES)
    
    # Main title
    fig.suptitle('Figure 5: Imputation Quality Comparison Across Missing Patterns',
                 fontsize=13, fontweight='bold', y=1.02)
    
    # Save
    output_pdf = os.path.join(output_dir, 'figure5_imputation.pdf')
    output_png = os.path.join(output_dir, 'figure5_imputation.png')
    
    plt.savefig(output_pdf, format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.savefig(output_png, format='png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"\n{'=' * 70}")
    print(f"Saved: {output_pdf}")
    print(f"Saved: {output_png}")
    print(f"{'=' * 70}")
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    
    for pattern_name in PATTERN_CONFIG.keys():
        print(f"\n{pattern_name}:")
        for model_name in MODEL_NAMES:
            if model_name in errors_by_pattern[pattern_name]:
                mae = np.mean(errors_by_pattern[pattern_name][model_name])
                print(f"  {model_name}: MAE = {mae:.4f}")
        
        print(f"  Boundary MAE:")
        for model_name in MODEL_NAMES:
            if model_name in boundary_errors_by_pattern[pattern_name]:
                bmae = boundary_errors_by_pattern[pattern_name][model_name]
                print(f"    {model_name}: {bmae:.4f}")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate Figure 5 for HELIX paper')
    parser.add_argument('--output_dir', type=str, 
                        default='/home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/imputation_visualization',
                        help='Output directory')
    args = parser.parse_args()
    
    generate_figure5(args.output_dir)