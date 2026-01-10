"""
Extract Feature Attention Weights from HELIX model
Simplified version - directly access saved attention weights (no hooks needed)

Requires the modified core.py with attention weight saving.

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

sys.path.insert(0, '/home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code')

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


def load_model(model_path, dataset_name, device):
    """Load trained HELIX model with attention saving capability."""
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


def load_data(data_path, n_samples=64):
    """Load sample data."""
    with h5py.File(data_path, 'r') as f:
        X = f['X'][:n_samples]
        X_ori = f['X_ori'][:n_samples]
    
    X = np.nan_to_num(X, nan=0.0)
    missing_mask = (~np.isnan(X_ori)).astype(np.float32)
    
    return X, missing_mask


def aggregate_attention_to_stations(attn, n_stations=12, n_features=11):
    """
    Aggregate feature-level attention to station-level.
    
    For feature attention: attn shape is [B*T, F, F] -> averaged to [F, F]
    Then aggregate [F, F] to [n_stations, n_stations]
    """
    # Average over batch*time if needed
    if len(attn.shape) == 3:
        attn = attn.mean(axis=0)  # [F, F]
    
    station_attn = np.zeros((n_stations, n_stations))
    for i in range(n_stations):
        for j in range(n_stations):
            i_s, i_e = i * n_features, (i + 1) * n_features
            j_s, j_e = j * n_features, (j + 1) * n_features
            station_attn[i, j] = attn[i_s:i_e, j_s:j_e].mean()
    
    return station_attn


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


def plot_attention_analysis(station_attn, geo_dist, stations, output_dir, layer_name="layer0"):
    """Generate attention analysis plots."""
    n = len(stations)
    short_names = [STATION_SHORT[s] for s in stations]
    
    # Compute geo proximity
    geo_prox = np.zeros_like(geo_dist)
    mask = geo_dist > 0
    geo_prox[mask] = 1.0 / geo_dist[mask]
    geo_prox = geo_prox / geo_prox.max()
    np.fill_diagonal(geo_prox, 1.0)
    
    # Correlation
    triu = np.triu_indices(n, k=1)
    r, p = pearsonr(geo_prox[triu], station_attn[triu])
    print(f"{layer_name} Attention vs Geographic Proximity: r = {r:.4f}, p = {p:.6f}")
    
    # === Combined heatmap ===
    fig, ax = plt.subplots(figsize=(8, 7))
    
    lower_mask = np.tril_indices(n, k=-1)
    upper_mask = np.triu_indices(n, k=1)
    
    geo_lower = geo_prox[lower_mask]
    geo_lower_norm = (geo_lower - geo_lower.min()) / (geo_lower.max() - geo_lower.min() + 1e-8)
    
    attn_upper = station_attn[upper_mask]
    attn_upper_norm = (attn_upper - attn_upper.min()) / (attn_upper.max() - attn_upper.min() + 1e-8)
    
    combined = np.full((n, n), np.nan)
    for idx, (i, j) in enumerate(zip(*lower_mask)):
        combined[i, j] = geo_lower_norm[idx]
    for idx, (i, j) in enumerate(zip(*upper_mask)):
        combined[i, j] = attn_upper_norm[idx]
    
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color='white')
    
    im = ax.imshow(combined, cmap=cmap, vmin=0, vmax=1)
    for i in range(n + 1):
        ax.axhline(i - 0.5, color='white', linewidth=0.5)
        ax.axvline(i - 0.5, color='white', linewidth=0.5)
    ax.plot([-0.5, n - 0.5], [-0.5, n - 0.5], 'k-', linewidth=1.5)
    
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_names, fontsize=10)
    ax.set_yticklabels(short_names, fontsize=10)
    
    plt.colorbar(im, ax=ax, shrink=0.8, label='Normalized Value')
    
    ax.text(0.25, 0.95, 'Upper: Learned Attention', transform=ax.transAxes,
           fontsize=10, fontweight='bold', ha='center', va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.text(0.75, 0.05, 'Lower: Geographic Proximity', transform=ax.transAxes,
           fontsize=10, fontweight='bold', ha='center', va='bottom',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    p_str = 'p < 0.01' if p < 0.01 else f'p = {p:.3f}'
    ax.set_title(f'Geographic Proximity vs Feature Attention ({layer_name})\n(Pearson r = {r:.2f}, {p_str})',
                fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'attention_{layer_name}_combined.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'attention_{layer_name}_combined.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return r, p


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--n_samples', type=int, default=64)
    args = parser.parse_args()
    
    MODEL_PATH = "/home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/reproduce_imputation/point01_log/BeijingAir_log/HELIX_BeijingAir_attn/round_0/HELIX.pypots"
    DATA_PATH = "/home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/data/generated_datasets/beijing_air_quality_rate01_step24_point/test.h5"
    OUTPUT_DIR = "/home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/attention_analysis"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 70)
    print("Feature Attention Analysis (Modified Model)")
    print("=" * 70)
    
    # Load model
    print("\nLoading model...")
    model = load_model(MODEL_PATH, 'BeijingAir', args.device)
    
    # Load data
    print("\nLoading data...")
    X, missing_mask = load_data(DATA_PATH, args.n_samples)
    print(f"Data shape: {X.shape}")
    
    # Forward pass
    print("\nRunning forward pass...")
    X_tensor = torch.tensor(X, dtype=torch.float32).to(args.device)
    mask_tensor = torch.tensor(missing_mask, dtype=torch.float32).to(args.device)
    
    with torch.no_grad():
        inputs = {'X': X_tensor, 'missing_mask': mask_tensor}
        _ = model.model.forward(inputs)
    
    # Get attention weights
    print("\nExtracting attention weights...")
    attn_dict = model.model.get_attention_weights()
    
    print(f"Available attention keys: {list(attn_dict.keys())}")
    
    # Compute geographic distance
    geo_dist = compute_geo_distance(STATION_ORDER, BEIJING_STATION_COORDS)
    
    # Analyze each layer's feature attention
    results = {}
    for key, attn in attn_dict.items():
        if 'feature' in key:
            print(f"\n{key}: shape {attn.shape}")
            
            # Convert to numpy
            attn_np = attn.cpu().numpy() if torch.is_tensor(attn) else attn
            
            # Save raw attention
            np.save(os.path.join(OUTPUT_DIR, f'{key}_raw.npy'), attn_np)
            
            # Aggregate to stations
            station_attn = aggregate_attention_to_stations(attn_np, N_STATIONS, N_FEATURES_PER_STATION)
            np.save(os.path.join(OUTPUT_DIR, f'{key}_station.npy'), station_attn)
            
            # Generate plots
            r, p = plot_attention_analysis(station_attn, geo_dist, STATION_ORDER, OUTPUT_DIR, key)
            results[key] = (r, p)
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    for key, (r, p) in results.items():
        p_str = 'p < 0.01' if p < 0.01 else f'p = {p:.4f}'
        print(f"{key}: r = {r:.4f}, {p_str}")
    print(f"\nOutput: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()