"""
Feature Identity Embedding Visualization for HELIX Paper Figure 2
ICML 2026 Submission - Nature/Science Style

Improvements:
1. Beijing administrative boundaries (16 districts) + Ring roads (2-6)
2. Blue-to-Red colormap for better visibility
3. Short labels + adjustText for auto-positioning
4. Publication-quality formatting (Times font, clean style)

Requirements:
    pip install matplotlib numpy torch scikit-learn scipy requests adjustText

Usage:
    python embedding_visualization.py

Author: Generated for HELIX ICML 2026 submission
"""

import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection, LineCollection
from sklearn.metrics.pairwise import cosine_similarity
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Try to import adjustText, provide fallback
try:
    from adjustText import adjust_text
    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False
    print("Warning: adjustText not installed. Labels may overlap.")
    print("Install with: pip install adjustText")

# Try to import requests for fetching GeoJSON
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("Warning: requests not installed. Will use simplified boundaries.")

# =============================================================================
# ICML/Nature/Science Style Configuration
# =============================================================================

plt.rcParams.update({
    # Font settings - ICML uses Times
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    
    # Figure settings
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    
    # Clean style
    'axes.linewidth': 0.8,
    'axes.edgecolor': 'black',
    'axes.facecolor': 'white',
    'axes.grid': False,
    'axes.spines.top': True,
    'axes.spines.right': True,
    
    # Ticks
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    
    # Lines
    'lines.linewidth': 1.0,
    
    # Legend
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': 'gray',
})

# =============================================================================
# Station Configuration
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

STATION_SHORT_CN = {
    'Huairou': '怀柔', 'Shunyi': '顺义', 'Wanliu': '万柳', 'Dingling': '定陵',
    'Nongzhanguan': '农展馆', 'Dongsi': '东四', 'Tiantan': '天坛',
    'Wanshouxigong': '万寿西宫', 'Aotizhongxin': '奥体中心', 'Changping': '昌平',
    'Guanyuan': '官园', 'Gucheng': '古城'
}

CN_LABELS = {
    'colorbar_sim': '嵌入相似度',
    'xlabel_lon': '经度 (°E)',
    'ylabel_lat': '纬度 (°N)',
    'title_a': '(a) 北京监测站点',
    'xlabel_dist': '地理距离 (km)',
    'ylabel_sim': '嵌入相似度',
    'title_b': '(b) 嵌入相似度与地理距离',
    'legend_fit': '线性拟合',
    'legend_ci': '95% 置信区间',
    'colorbar_norm': '归一化数值',
    'upper_label': '上三角: 学习嵌入',
    'lower_label': '下三角: 地理',
    'title_c_prefix': '(c) 地理 vs 学习嵌入',
}

N_STATIONS = 12
N_FEATURES_PER_STATION = 11

# Beijing Ring Road approximate coordinates (GCJ-02)
# These are simplified center points and radii
RING_ROADS = {
    '2nd Ring': {'center': (116.397, 39.908), 'radius_km': 3.5},
    '3rd Ring': {'center': (116.397, 39.920), 'radius_km': 7.0},
    '4th Ring': {'center': (116.397, 39.930), 'radius_km': 10.5},
    '5th Ring': {'center': (116.397, 39.940), 'radius_km': 14.0},
    '6th Ring': {'center': (116.397, 39.960), 'radius_km': 22.0},
}

# =============================================================================
# GeoJSON Data Fetching
# =============================================================================

def fetch_beijing_geojson(api_key=None, use_cache=True, cache_file='beijing_districts.json'):
    """
    Fetch Beijing district boundaries from Amap DataV API.
    Returns GeoJSON FeatureCollection with 16 districts.
    """
    if use_cache and os.path.exists(cache_file):
        print(f"Loading cached GeoJSON from {cache_file}")
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    if not HAS_REQUESTS:
        print("requests not available, returning None")
        return None
    
    # Amap DataV public API (no key needed)
    url = "https://geo.datav.aliyun.com/areas_v3/bound/110000_full.json"
    
    try:
        print(f"Fetching Beijing GeoJSON from {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        geojson = response.json()
        
        # Cache for future use
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(geojson, f, ensure_ascii=False)
        print(f"Cached to {cache_file}")
        
        return geojson
    except Exception as e:
        print(f"Failed to fetch GeoJSON: {e}")
        return None


def parse_geojson_boundaries(geojson):
    """
    Parse GeoJSON to extract district boundaries.
    Returns dict: {district_name: list of (lon, lat) polygons}
    """
    if geojson is None:
        return {}
    
    boundaries = {}
    for feature in geojson.get('features', []):
        props = feature.get('properties', {})
        name = props.get('name', 'Unknown')
        geometry = feature.get('geometry', {})
        geom_type = geometry.get('type', '')
        coords = geometry.get('coordinates', [])
        
        polygons = []
        if geom_type == 'Polygon':
            # Single polygon - take outer ring
            if coords:
                polygons.append(np.array(coords[0]))
        elif geom_type == 'MultiPolygon':
            # Multiple polygons
            for poly in coords:
                if poly:
                    polygons.append(np.array(poly[0]))
        
        if polygons:
            boundaries[name] = polygons
    
    return boundaries


def create_ring_road_circle(center_lon, center_lat, radius_km, n_points=100):
    """
    Create approximate circle coordinates for ring roads.
    Note: This is a simplified version; real ring roads are not perfect circles.
    """
    # Approximate conversion at Beijing's latitude
    km_per_deg_lon = 111.32 * np.cos(np.radians(center_lat))
    km_per_deg_lat = 110.574
    
    angles = np.linspace(0, 2 * np.pi, n_points)
    lons = center_lon + (radius_km / km_per_deg_lon) * np.cos(angles)
    lats = center_lat + (radius_km / km_per_deg_lat) * np.sin(angles)
    
    return np.column_stack([lons, lats])


# =============================================================================
# Model Loading Functions
# =============================================================================

def load_feature_embedding(model_path):
    """Load Feature Identity Embedding from trained HELIX model."""
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    embedding = state_dict['backbone.embedding.feature_id'].numpy()
    print(f"Loaded embedding with shape: {embedding.shape}")
    return embedding


def compute_station_similarity(embedding, n_stations=12, n_features=11):
    """
    Compute station-level similarity by comparing SAME features across stations.
    """
    embed_dim = embedding.shape[1]
    embedding_3d = embedding.reshape(n_stations, n_features, embed_dim)
    
    station_sim = np.zeros((n_stations, n_stations))
    
    for i in range(n_stations):
        for j in range(n_stations):
            if i == j:
                station_sim[i, j] = 1.0
            else:
                feature_sims = []
                for k in range(n_features):
                    emb_i_k = embedding_3d[i, k]
                    emb_j_k = embedding_3d[j, k]
                    cos_sim = np.dot(emb_i_k, emb_j_k) / (np.linalg.norm(emb_i_k) * np.linalg.norm(emb_j_k) + 1e-8)
                    feature_sims.append(cos_sim)
                station_sim[i, j] = np.mean(feature_sims)
    
    return station_sim


def compute_geographic_distance_matrix(stations, coords_dict):
    """Compute pairwise geographic distance (in km) using Haversine formula."""
    from math import radians, cos, sin, asin, sqrt
    
    def haversine(lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon, dlat = lon2 - lon1, lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        return 2 * asin(sqrt(a)) * 6371
    
    n = len(stations)
    dist_matrix = np.zeros((n, n))
    for i, s1 in enumerate(stations):
        for j, s2 in enumerate(stations):
            if i != j:
                dist_matrix[i, j] = haversine(
                    coords_dict[s1]['longitude'], coords_dict[s1]['latitude'],
                    coords_dict[s2]['longitude'], coords_dict[s2]['latitude']
                )
    return dist_matrix


# =============================================================================
# Visualization Functions
# =============================================================================
def create_figure2(similarity_matrix, distance_matrix, stations, 
                            output_path, district_boundaries=None, top_n=20, lang='en'):
    """
    Create combined Figure 2 with 3 subplots in HORIZONTAL layout: 
    (a) Map, (b) Scatter, (c) Heatmap
    
    This layout is suitable for single-column figures in ICML papers.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.gridspec as gridspec
    
    # Select labels based on language
    if lang == 'cn':
        L = CN_LABELS
        short_name_map = STATION_SHORT_CN
    else:
        L = {
            'colorbar_sim': 'Embedding Similarity',
            'xlabel_lon': 'Longitude (°E)',
            'ylabel_lat': 'Latitude (°N)',
            'title_a': '(a) Beijing Stations',
            'xlabel_dist': 'Geographic Distance (km)',
            'ylabel_sim': 'Embedding Similarity',
            'title_b': '(b) Embedding vs. Geography',
            'legend_fit': 'Linear fit',
            'legend_ci': '95% CI',
            'colorbar_norm': 'Normalized Value',
            'upper_label': 'Upper: Learned',
            'lower_label': 'Lower: Geo.',
            'title_c_prefix': '(c) Geo. vs Learned',
        }
        short_name_map = STATION_SHORT
    fig = plt.figure(figsize=(6.5, 2.8))  # 减少高度，b会自动变扁
    outer_gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.1, 2.5],
                wspace=0.15, left=0.02, right=0.96,
                top=0.88, bottom=0.32)  # 增加底部空间给colorbar
    
    # a单独占左边
    ax1 = fig.add_subplot(outer_gs[0, 0])
    
    # b和c用内层GridSpec，间距更小
    inner_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_gs[0, 1],
                                 wspace=0.25, width_ratios=[1, 1])
    ax2 = fig.add_subplot(inner_gs[0, 0])
    ax3 = fig.add_subplot(inner_gs[0, 1])
    
    lons = np.array([BEIJING_STATION_COORDS[s]['longitude'] for s in stations])
    lats = np.array([BEIJING_STATION_COORDS[s]['latitude'] for s in stations])
    n = len(stations)
    short_names = [short_name_map[s] for s in stations]
    
    # ===================
    # (a) Beijing Map
    # ===================
    
    # Plot district boundaries
    if district_boundaries:
        for name, polygons in district_boundaries.items():
            for poly in polygons:
                if len(poly) > 0:
                    ax1.plot(poly[:, 0], poly[:, 1], color='#CCCCCC', 
                            linewidth=0.5, zorder=1)
    
    # Ring roads
    ring_colors = ['#E8E8E8', '#E0E0E0', '#D8D8D8', '#D0D0D0', '#C8C8C8']
    for idx, (ring_name, ring_info) in enumerate(RING_ROADS.items()):
        ring_coords = create_ring_road_circle(
            ring_info['center'][0], ring_info['center'][1], 
            ring_info['radius_km'], n_points=80
        )
        ax1.plot(ring_coords[:, 0], ring_coords[:, 1], 
                color=ring_colors[idx], linewidth=0.6, 
                linestyle='--', alpha=0.5, zorder=1)
    
    # Connections
    connections = []
    for i in range(n):
        for j in range(i+1, n):
            connections.append((i, j, similarity_matrix[i, j]))
    connections.sort(key=lambda x: x[2], reverse=True)
    top_connections = connections[:top_n]
    
    sim_values = [c[2] for c in top_connections]
    sim_min, sim_max = min(sim_values), max(sim_values)
    
    cmap = plt.cm.coolwarm
    norm = Normalize(vmin=sim_min, vmax=sim_max)
    
    for i, j, sim in top_connections:
        color = cmap(norm(sim))
        # 增加非线性宽度和透明度，突出核心拓扑
        width = 0.3 + 1.2 * ((sim - sim_min) / (sim_max - sim_min))**2 
        alpha = 0.4 + 0.6 * ((sim - sim_min) / (sim_max - sim_min))
        ax1.plot([lons[i], lons[j]], [lats[i], lats[j]], 
                color=color, linewidth=width, alpha=alpha, zorder=2)
    
    # Stations
    ax1.scatter(lons, lats, c='white', s=60, edgecolors='black',  # s从250改为150
       linewidths=0.3, zorder=4)
    
    # Labels
    for i, station in enumerate(stations):
        ax1.annotate(short_name_map[station], (lons[i], lats[i]), 
                fontsize=4, ha='center', va='center',  # 6改为5
                fontweight='bold', zorder=5)
    
    # Colorbar for map
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax1 = inset_axes(ax1, width="75%", height="4%", loc='lower center',
                  bbox_to_anchor=(0, -0.3, 1, 1), bbox_transform=ax1.transAxes)
    cbar1 = fig.colorbar(sm, cax=cax1, orientation='horizontal')
    cbar1.set_label(L['colorbar_sim'], fontsize=7, labelpad=1)
    cbar1.ax.tick_params(labelsize=6)
    
    ax1.set_xlabel(L['xlabel_lon'], fontsize=8, labelpad=1)  # 减少间距
    ax1.set_ylabel(L['ylabel_lat'], fontsize=8, labelpad=1)  # 减少留白
    ax1.set_title(L['title_a'], fontsize=10)  # 简化标题
    ax1.set_xlim(116.10, 116.72)
    ax1.set_ylim(39.82, 40.38)
    ax1.tick_params(axis='both', labelsize=7)  # 添加这行
    ax1.set_facecolor('#FAFAFA')
    ax1.set_aspect('auto')
    
    # ===================
    # (b) Scatter Plot - 恢复colorbar
    # ===================
    
    triu_idx = np.triu_indices(n, k=1)
    distances = distance_matrix[triu_idx]
    similarities = similarity_matrix[triu_idx]
    
    r, p = pearsonr(distances, similarities)
    
    # 恢复颜色映射
    scatter = ax2.scatter(distances, similarities, c=distances, cmap='viridis_r',
                         alpha=0.75, s=45, edgecolors='white', linewidths=0.4)
    
    z = np.polyfit(distances, similarities, 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(distances.min(), distances.max(), 100)
    ax2.plot(x_line, p_line(x_line), 'r-', linewidth=1.5, label=L['legend_fit'])
    
    residuals = similarities - p_line(distances)
    std_err = np.std(residuals)
    ax2.fill_between(x_line, p_line(x_line) - 1.96*std_err, 
                    p_line(x_line) + 1.96*std_err, 
                    color='red', alpha=0.1, label=L['legend_ci'])
    
    ax2.set_xlabel(L['xlabel_dist'], fontsize=9)
    ax2.set_ylabel(L['ylabel_sim'], fontsize=8, labelpad=0)  # labelpad增大使标签右移
    ax2.tick_params(axis='y', labelsize=7, pad=1)
    p_str = f'p = {p:.4f}' if p >= 0.0001 else 'p < 0.0001'
    ax2.text(0.05, 0.05, f'$r = {r:.3f}$\n$p < 0.0001$', 
         transform=ax2.transAxes, fontsize=8, 
         verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    ax2.set_title(L['title_b'], fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.4)
    ax2.legend(loc='upper right', fontsize=6, framealpha=0.9)
    ymin, ymax = ax2.get_ylim()
    ax2.set_ylim(ymin - 0.02, ymax + 0.02)
    
    # ===================
    # (c) Combined Heatmap - colorbar手动定位
    # ===================
    
    # Compute normalized values
    geo_proximity = np.zeros_like(distance_matrix)
    mask = distance_matrix > 0
    geo_proximity[mask] = 1.0 / distance_matrix[mask]
    
    lower_mask = np.tril_indices(n, k=-1)
    geo_lower = geo_proximity[lower_mask]
    geo_lower_norm = (geo_lower - geo_lower.min()) / (geo_lower.max() - geo_lower.min())
    
    upper_mask = np.triu_indices(n, k=1)
    emb_upper = similarity_matrix[upper_mask]
    emb_upper_norm = (emb_upper - emb_upper.min()) / (emb_upper.max() - emb_upper.min()) * 0.7
    
    combined = np.full((n, n), np.nan)
    for idx, (i, j) in enumerate(zip(*lower_mask)):
        combined[i, j] = geo_lower_norm[idx]
    for idx, (i, j) in enumerate(zip(*upper_mask)):
        combined[i, j] = emb_upper_norm[idx]
    
    cmap_heat = plt.cm.YlOrRd.copy()
    cmap_heat.set_bad(color='white')
    
    im = ax3.imshow(combined, cmap=cmap_heat, vmin=0, vmax=1, aspect='auto')
    ax3.set_anchor('N')
    
    for i in range(n + 1):
        ax3.axhline(i - 0.5, color='white', linewidth=0.4)
        ax3.axvline(i - 0.5, color='white', linewidth=0.4)
    ax3.plot([-0.5, n - 0.5], [-0.5, n - 0.5], 'k-', linewidth=1.0)
    
    ax3.set_xticks(range(n))
    ax3.set_yticks(range(n))
    _heat_fs = 5.5 if lang == 'cn' else 8
    ax3.set_xticklabels(short_names, fontsize=_heat_fs, rotation=45, ha='right', rotation_mode='anchor')
    ax3.set_yticklabels(short_names, fontsize=_heat_fs)
    
    # Colorbar - 使用fig.colorbar手动定位，避免从subplot借空间
    # 获取ax3的位置后，在其右侧添加colorbar
    cax3 = inset_axes(ax3, width="75%", height="4%", loc='lower center',
                  bbox_to_anchor=(0, -0.3, 1, 1), bbox_transform=ax3.transAxes)
    cbar3 = fig.colorbar(im, cax=cax3, orientation='horizontal')
    cbar3.set_label(L['colorbar_norm'], fontsize=7, labelpad=1)
    cbar3.ax.tick_params(labelsize=6)
    
    # Annotations
    ax3.text(0.25, 0.92, L['upper_label'], transform=ax3.transAxes, 
            fontsize=6, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='wheat', alpha=0.8))
    ax3.text(0.75, 0.08, L['lower_label'], transform=ax3.transAxes,  # 缩短文字
            fontsize=6, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', alpha=0.8))
    
    # Correlation for heatmap title
    r_heat, p_heat = pearsonr(geo_proximity[triu_idx], similarity_matrix[triu_idx])
    ax3.set_title(f'{L["title_c_prefix"]} (r = {r_heat:.3f})', fontsize=9)  # 简化
    
    plt.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0.05)
    plt.savefig(output_path.replace('.pdf', '.png'), format='png', dpi=300, bbox_inches='tight', pad_inches=0.05)
    print(f"Saved: {output_path}")
    plt.close()
    
    return r, p


# =============================================================================
# Main Function
# =============================================================================

def main():
    """Main function to generate all visualizations."""
    
    # Configuration - MODIFY THESE PATHS AS NEEDED
    MODEL_PATH = "/home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/reproduce_imputation/point01_log/BeijingAir_log/HELIX_BeijingAir/round_0/20260103_T180923/HELIX.pypots"
    OUTPUT_DIR = "/home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/embedding_analysis"
    TOP_N_CONNECTIONS = 25
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 70)
    print("Feature Identity Embedding Analysis - ICML Style (horizontal LAYOUT)")
    print("=" * 70)
    
    # Fetch Beijing district boundaries
    cache_file = os.path.join(OUTPUT_DIR, "beijing_districts.json")
    geojson = fetch_beijing_geojson(use_cache=True, cache_file=cache_file)
    district_boundaries = parse_geojson_boundaries(geojson)
    
    if district_boundaries:
        print(f"Loaded {len(district_boundaries)} district boundaries")
    else:
        print("No district boundaries available, using simplified map")
    
    # Load model and compute similarity
    embedding = load_feature_embedding(MODEL_PATH)
    
    print("\nComputing station similarity...")
    similarity_matrix = compute_station_similarity(embedding, N_STATIONS, N_FEATURES_PER_STATION)
    print(f"Similarity range: [{similarity_matrix.min():.3f}, {similarity_matrix.max():.3f}]")
    
    distance_matrix = compute_geographic_distance_matrix(STATION_ORDER, BEIJING_STATION_COORDS)
    
    print(f"\nGenerating visualizations...")
    print(f"Output directory: {OUTPUT_DIR}")
    
    print("\n4. Generating horizontal combined figure...")
    r3, p3 = create_figure2(similarity_matrix, distance_matrix, STATION_ORDER,
                                     os.path.join(OUTPUT_DIR, "figure2.pdf"),
                                     district_boundaries=district_boundaries,
                                     top_n=TOP_N_CONNECTIONS)
    # === Chinese version ===
    import matplotlib as mpl
    _orig_family = mpl.rcParams['font.family']
    _orig_serif = mpl.rcParams['font.serif']
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC']
    mpl.rcParams['axes.unicode_minus'] = False

    print("\n5. Generating Chinese version...")
    create_figure2(similarity_matrix, distance_matrix, STATION_ORDER,
                   os.path.join(OUTPUT_DIR, "figure2_cn.pdf"),
                   district_boundaries=district_boundaries,
                   top_n=TOP_N_CONNECTIONS, lang='cn')

    # Restore font settings
    mpl.rcParams['font.family'] = _orig_family
    mpl.rcParams['font.serif'] = _orig_serif
    mpl.rcParams['axes.unicode_minus'] = True
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Pearson correlation (embedding similarity vs geographic distance): r = {r3:.3f}")
    print(f"p-value: {p3:.6f}")
    print(f"\nGenerated files:")
    # print(f"  - {OUTPUT_DIR}/beijing_map.pdf (+ .png)")
    # print(f"  - {OUTPUT_DIR}/beijing_scatter.pdf (+ .png)")
    # print(f"  - {OUTPUT_DIR}/beijing_heatmap.pdf (+ .png)")
    print(f"  - {OUTPUT_DIR}/figure2.pdf (+ .png) <- horizontal figure for paper")
    print(f"  - {OUTPUT_DIR}/figure2_cn.pdf (+ .png) <- 中文版")
    print("=" * 70)


if __name__ == "__main__":
    main()
