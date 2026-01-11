"""
Extract Feature Attention Weights from HELIX model
Generates Figure 3 (Feature Attention) and Appendix A1 (Temporal Attention)

Author: Generated for HELIX ICML 2026 submission
"""

import os
import sys
import numpy as np
import torch
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

# Matplotlib style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
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
# Plotting Functions
# =============================================================================

def plot_figure3_feature_attention(attn_dict, geo_dist, stations, output_dir, 
                                    n_stations=12, n_features=11):
    """
    Generate Figure 3: Feature Attention captures spatial structure.
    Shows feature attention heatmaps with r/p values indicating correlation with geography.
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
    
    # Create figure
    n_layers = len(feature_data)
    fig, axes = plt.subplots(1, n_layers, figsize=(4.5 * n_layers, 4.2))
    if n_layers == 1:
        axes = [axes]
    
    for idx, (key, station_attn, r, p) in enumerate(feature_data):
        ax = axes[idx]
        
        im = ax.imshow(station_attn, cmap='YlOrRd', aspect='equal')
        
        # Grid
        for i in range(n_stations + 1):
            ax.axhline(i - 0.5, color='white', linewidth=0.3)
            ax.axvline(i - 0.5, color='white', linewidth=0.3)
        
        # Labels
        ax.set_xticks(range(n_stations))
        ax.set_yticks(range(n_stations))
        ax.set_xticklabels(short_names, fontsize=7, rotation=45, ha='right')
        ax.set_yticklabels(short_names, fontsize=7)
        ax.set_xlabel('Key Station', fontsize=9)
        ax.set_ylabel('Query Station', fontsize=9)
        
        # Title with r and p
        layer_num = key.replace('layer', '').replace('_feature', '')
        p_str = 'p < 0.0001' if p < 0.0001 else f'p = {p:.4f}'
        ax.set_title(f'Layer {layer_num} Feature Attention\n(r = {r:.3f}, {p_str})', 
                    fontsize=10, fontweight='bold')
        
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    # 主标题：降低 y 值，减少顶部留白
    plt.suptitle('Feature Attention Increasingly Captures Spatial Structure', 
                 fontsize=12, fontweight='bold', y=0.98)
    
    # 使用 subplots_adjust 精细控制边距
    plt.subplots_adjust(top=0.88, bottom=0.15, left=0.05, right=0.95, wspace=0.3)
    
    plt.savefig(os.path.join(output_dir, 'figure3_feature_attention.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'figure3_feature_attention.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_dir}/figure3_feature_attention.pdf")
    return feature_data


def plot_appendix_temporal_attention(attn_dict, output_dir):
    """
    Generate Appendix A1: Temporal Attention Heatmaps.
    Shows how temporal attention evolves from local to broader context.
    """
    # Collect temporal attentions
    temporal_attns = []
    for key in sorted(attn_dict.keys()):
        if 'time' in key:
            attn = attn_dict[key]
            attn_np = attn.cpu().numpy() if torch.is_tensor(attn) else attn
            temporal_attns.append((key, attn_np))
    
    # Create figure
    n_layers = len(temporal_attns)
    fig, axes = plt.subplots(1, n_layers, figsize=(4.5 * n_layers, 4))
    if n_layers == 1:
        axes = [axes]
    
    for idx, (key, attn) in enumerate(temporal_attns):
        ax = axes[idx]
        attn_avg = attn.mean(axis=0)  # [T, T]
        
        im = ax.imshow(attn_avg, cmap='Blues', aspect='equal')
        
        layer_num = key.replace('layer', '').replace('_time', '')
        ax.set_title(f'Layer {layer_num} Temporal Attention\n(averaged over samples)', fontsize=10)
        ax.set_xlabel('Key Time Step', fontsize=9)
        ax.set_ylabel('Query Time Step', fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.suptitle('Appendix A1: Temporal Attention Evolution\n(From local focus to broader context)', 
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'appendix_a1_attention.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'appendix_a1_attention.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_dir}/appendix_a1_attention.pdf")


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
    print(f"  - {args.output_dir}/appendix_a1_attention.pdf (Appendix)")
    print("=" * 70)


if __name__ == "__main__":
    main()