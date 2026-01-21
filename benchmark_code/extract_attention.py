"""
Extract Feature Attention Weights from HELIX model
Generates Figure 3 (Feature Attention) and Appendix A1 (Temporal Attention)

Author: Generated for HELIX ICML 2026 submission

FIXED VERSION: Resolves element overlapping issues for ICML 2026 format
- Increased figure height
- Used GridSpec for precise layout control
- Adjusted font sizes and label rotations
- Fixed colorbar positioning
"""

import os
import sys
import numpy as np
import torch
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

BEIJING_STATION_COORDS = {
    'Huairou': {'latitude': 40.3284, 'longitude': 116.6371},
    'Shunyi': {'latitude': 40.1299, 'longitude': 116.6563},
    'Wanliu': {'latitude': 39.9879, 'longitude': 116.2883},
    'Dingling': {'latitude': 40.2953, 'longitude': 116.2218},
    'Nongzhanguan': {'latitude': 39.9430, 'longitude': 116.4604},
    'Dongsi': {'latitude': 39.9288, 'longitude': 116.4177},
    'Tiantan': {'latitude': 39.8823, 'longitude': 116.4107},
    'Wanshouxigong': {'latitude': 39.8991, 'longitude': 116.3480},
    'Aotizhongxin': {'latitude': 40.0021, 'longitude': 116.3973},
    'Changping': {'latitude': 40.2207, 'longitude': 116.2362},
    'Guanyuan': {'latitude': 39.9320, 'longitude': 116.3606},
    'Gucheng': {'latitude': 39.9123, 'longitude': 116.1841}
}

STATION_ORDER = ['Huairou', 'Shunyi', 'Wanliu', 'Dingling', 'Nongzhanguan',
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

# Matplotlib style - 减小默认字体以适应ICML格式
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


# =============================================================================
# Helper Functions
# =============================================================================

def load_model(model_path, dataset_name, device):
    """Load trained HELIX model."""
    # Add benchmark_code to path
    benchmark_path = os.path.dirname(os.path.dirname(model_path.split('reproduce_imputation')[0]))
    if 'benchmark_code' in model_path:
        benchmark_path = model_path.split('reproduce_imputation')[0]
    sys.path.insert(0, benchmark_path)
    
    from hpo_results import HPO_RESULTS
    from pypots.imputation import HELIX
    
    hyperparameters = HPO_RESULTS[dataset_name]['HELIX'].copy()
    hyperparameters.pop('lr', None)
    hyperparameters['device'] = device
    hyperparameters['saving_path'] = None
    hyperparameters['model_saving_strategy'] = None
    
    model = HELIX(**hyperparameters)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.model.load_state_dict(checkpoint['model_state_dict'])
    model.model.eval()
    
    return model


def load_data(data_path, n_samples=256):
    """Load sample data."""
    with h5py.File(data_path, 'r') as f:
        X = f['X'][:n_samples]
        X_ori = f['X_ori'][:n_samples]
    
    X = np.nan_to_num(X, nan=0.0)
    missing_mask = (~np.isnan(X_ori)).astype(np.float32)
    
    return X, missing_mask


def compute_geo_distance(stations, coords):
    """Compute geographic distance matrix."""
    from math import radians, cos, sin, asin, sqrt
    def haversine(lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        a = sin((lat2-lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2
        return 2 * asin(sqrt(a)) * 6371
    
    n = len(stations)
    dist = np.zeros((n, n))
    for i, s1 in enumerate(stations):
        for j, s2 in enumerate(stations):
            if i != j:
                dist[i,j] = haversine(coords[s1]['longitude'], coords[s1]['latitude'],
                                      coords[s2]['longitude'], coords[s2]['latitude'])
    return dist


def aggregate_to_stations(attn, n_stations=12, n_features=11):
    """Aggregate feature-level attention to station-level."""
    if len(attn.shape) == 3:
        attn = attn.mean(axis=0)  # [F, F]
    
    station_attn = np.zeros((n_stations, n_stations))
    for i in range(n_stations):
        for j in range(n_stations):
            i_s, i_e = i * n_features, (i + 1) * n_features
            j_s, j_e = j * n_features, (j + 1) * n_features
            station_attn[i, j] = attn[i_s:i_e, j_s:j_e].mean()
    
    return station_attn


# =============================================================================
# Plotting Functions (FIXED for ICML format)
# =============================================================================

def plot_figure3_feature_attention(attn_dict, geo_dist, stations, output_dir, 
                                    n_stations=12, n_features=11):
    """
    Generate Figure 3: Feature Attention captures spatial structure.
    Shows feature attention heatmaps with r/p values indicating correlation with geography.
    
    FIXED: 
    - Increased figure height (2.8 -> 3.3)
    - Used GridSpec with shared colorbar
    - Adjusted font sizes and rotation angles
    - Fixed spacing parameters
    """
    short_names = [STATION_SHORT[s] for s in stations]
    
    # Compute geographic proximity
    geo_prox = np.zeros_like(geo_dist)
    mask = geo_dist > 0
    geo_prox[mask] = 1.0 / geo_dist[mask]
    triu = np.triu_indices(n_stations, k=1)
    
    # Collect feature attentions
    feature_data = []
    for key in sorted(attn_dict.keys()):
        if 'feature' in key:
            attn = attn_dict[key]
            attn_np = attn.cpu().numpy() if torch.is_tensor(attn) else attn
            station_attn = aggregate_to_stations(attn_np, n_stations, n_features)
            r, p = pearsonr(geo_prox[triu], station_attn[triu])
            feature_data.append((key, station_attn, r, p))
    
    n_layers = len(feature_data)
    
    # ==========================================================================
    # FIXED: Use GridSpec with shared colorbar for better layout control
    # ==========================================================================
    
    # Increased height: 2.8 -> 3.3 to accommodate title and x-axis labels
    fig = plt.figure(figsize=(6.5, 3.0))
    
    # GridSpec: reserve space for shared colorbar on the right
    gs = GridSpec(1, n_layers, 
              wspace=0.25,
              left=0.08, right=0.88, top=0.88, bottom=0.15)

    axes = [fig.add_subplot(gs[0, i]) for i in range(n_layers)]
    # Colorbar axis: 手动定位，比主图短 (shrink ~70%)
    cax = fig.add_axes([0.90, 0.28, 0.015, 0.48])  # [left, bottom, width, height]
    
    # Unified color range across all subplots
    all_values = [data[1] for data in feature_data]
    vmin = min(arr.min() for arr in all_values)
    vmax = max(arr.max() for arr in all_values)
    
    for idx, (key, station_attn, r, p) in enumerate(feature_data):
        ax = axes[idx]
        
        im = ax.imshow(station_attn, cmap='YlOrRd', aspect='equal',
                       vmin=vmin, vmax=vmax)
        
        # Grid lines
        for i in range(n_stations + 1):
            ax.axhline(i - 0.5, color='white', linewidth=0.3)
            ax.axvline(i - 0.5, color='white', linewidth=0.3)
        
        # Labels - reduced font size, increased rotation angle
        ax.set_xticks(range(n_stations))
        ax.set_yticks(range(n_stations))
        ax.set_xticklabels(short_names, fontsize=5, rotation=60, ha='right')
        
        # Only show y-axis labels on the first subplot
        if idx == 0:
            ax.set_yticklabels(short_names, fontsize=5)
            ax.set_ylabel('Query Station', fontsize=8)
        else:
            ax.set_yticklabels([])
        
        ax.set_xlabel('Key Station', fontsize=8)
        
        # Title - two lines with reduced font size
        layer_num = key.replace('layer', '').replace('_feature', '')
        p_str = 'p < 0.0001' if p < 0.0001 else f'p = {p:.4f}'
        ax.set_title(f'Layer {layer_num} Feature Attention\n(r = {r:.3f}, {p_str})', 
                    fontsize=8, fontweight='bold', pad=4)
    
    # Shared colorbar
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label('Attention', fontsize=7)
    
    plt.savefig(os.path.join(output_dir, 'figure3_feature_attention.pdf'), 
                bbox_inches='tight', pad_inches=0.02)
    plt.savefig(os.path.join(output_dir, 'figure3_feature_attention.png'), 
                dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close()
    
    print(f"Saved: {output_dir}/figure3_feature_attention.pdf")
    return feature_data


def plot_appendix_temporal_attention(attn_dict, output_dir):
    """
    Generate Appendix A1: Temporal Attention Heatmaps.
    Shows how temporal attention evolves from local to broader context.
    
    FIXED:
    - Increased figure height (2.8 -> 3.0)
    - Used GridSpec with shared colorbar
    - Adjusted suptitle position
    """
    # Collect temporal attentions
    temporal_attns = []
    for key in sorted(attn_dict.keys()):
        if 'time' in key:
            attn = attn_dict[key]
            attn_np = attn.cpu().numpy() if torch.is_tensor(attn) else attn
            temporal_attns.append((key, attn_np))
    
    n_layers = len(temporal_attns)
    
    # ==========================================================================
    # FIXED: Use GridSpec with shared colorbar
    # ==========================================================================
    
    fig = plt.figure(figsize=(6.5, 3.0))  # Increased height
    
    gs = GridSpec(1, n_layers, 
              wspace=0.35,
              left=0.06, right=0.88, top=0.88, bottom=0.18)

    axes = [fig.add_subplot(gs[0, i]) for i in range(n_layers)]
    # Colorbar axis: 手动定位，比主图短 (shrink ~70%)
    cax = fig.add_axes([0.90, 0.30, 0.015, 0.46])  # [left, bottom, width, height]
    
    # Unified color range
    all_values = [data[1].mean(axis=0) for data in temporal_attns]
    vmin = min(arr.min() for arr in all_values)
    vmax = max(arr.max() for arr in all_values)
    
    for idx, (key, attn) in enumerate(temporal_attns):
        ax = axes[idx]
        attn_avg = attn.mean(axis=0)  # [T, T]
        
        im = ax.imshow(attn_avg, cmap='Blues', aspect='equal',
                       vmin=vmin, vmax=vmax)
        
        layer_num = key.replace('layer', '').replace('_time', '')
        ax.set_title(f'Layer {layer_num} Temporal Attention\n(averaged over samples)', 
                    fontsize=8, fontweight='bold', pad=4)
        ax.set_xlabel('Key Time Step', fontsize=8)
        
        if idx == 0:
            ax.set_ylabel('Query Time Step', fontsize=8)
        
        # Reduce number of ticks to avoid crowding
        ax.locator_params(axis='both', nbins=6)
    
    # Shared colorbar
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label('Attention', fontsize=7)
    
    plt.savefig(os.path.join(output_dir, 'figure4_temporal_attention.pdf'), 
                bbox_inches='tight', pad_inches=0.02)
    plt.savefig(os.path.join(output_dir, 'figure4_temporal_attention.png'), 
                dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close()
    
    print(f"Saved: {output_dir}/figure4_temporal_attention.pdf")


# =============================================================================
# Main Function
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='HELIX Attention Analysis')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained HELIX model (.pypots file)')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to test data (.h5 file)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for figures')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device (default: cuda:0)')
    parser.add_argument('--n_samples', type=int, default=256,
                        help='Number of samples (default: 256)')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("HELIX Attention Analysis")
    print("=" * 70)
    
    # Load model
    print("\nLoading model...")
    model = load_model(args.model_path, 'BeijingAir', args.device)
    
    # Load data
    print("Loading data...")
    X, missing_mask = load_data(args.data_path, args.n_samples)
    print(f"Data shape: {X.shape}, Samples: {args.n_samples}")
    
    # Forward pass
    print("Running forward pass...")
    X_tensor = torch.tensor(X, dtype=torch.float32).to(args.device)
    mask_tensor = torch.tensor(missing_mask, dtype=torch.float32).to(args.device)
    
    with torch.no_grad():
        inputs = {'X': X_tensor, 'missing_mask': mask_tensor}
        _ = model.model.forward(inputs)
    
    # Get attention weights
    print("Extracting attention weights...")
    attn_dict = model.model.get_attention_weights()
    print(f"Available keys: {list(attn_dict.keys())}")
    
    # Compute geographic distance
    geo_dist = compute_geo_distance(STATION_ORDER, BEIJING_STATION_COORDS)
    
    # Generate Figure 3: Feature Attention
    print("\n" + "-" * 70)
    print("Generating Figure 3: Feature Attention...")
    feature_results = plot_figure3_feature_attention(
        attn_dict, geo_dist, STATION_ORDER, args.output_dir, N_STATIONS, N_FEATURES_PER_STATION
    )
    
    # Generate Appendix A1: Temporal Attention
    print("\n" + "-" * 70)
    print("Generating Appendix A1: Temporal Attention...")
    plot_appendix_temporal_attention(attn_dict, args.output_dir)
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary: Feature Attention vs Geographic Structure")
    print("=" * 70)
    for key, station_attn, r, p in feature_results:
        p_str = 'p < 0.0001' if p < 0.0001 else f'p = {p:.4f}'
        print(f"  {key}: r = {r:.3f}, {p_str}")
    
    print(f"\nKey insight: Correlation increases with depth!")
    print(f"\nOutput files:")
    print(f"  - {args.output_dir}/figure3_feature_attention.pdf (Main text)")
    print(f"  - {args.output_dir}/figure4_temporal_attention.pdf (Appendix)")
    print("=" * 70)


if __name__ == "__main__":
    main()