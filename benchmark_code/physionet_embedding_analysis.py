"""
PhysioNet2012 Feature Embedding Analysis - Figure 6
Only generates Boxplot for within-group vs between-group comparison

Author: Generated for HELIX ICML 2026 submission
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

PHYSIONET_FEATURES = [
    "DiasABP", "HR", "Na", "Lactate", "NIDiasABP", "PaO2", "WBC", "pH",
    "Albumin", "ALT", "Glucose", "SaO2", "Temp", "AST", "Bilirubin", "HCO3",
    "BUN", "RespRate", "Mg", "HCT", "SysABP", "FiO2", "K", "GCS", "Cholesterol",
    "NISysABP", "TroponinT", "MAP", "TroponinI", "PaCO2", "Platelets", "Urine",
    "NIMAP", "Creatinine", "ALP"
]

FEATURE_GROUPS = {
    'Blood Pressure': ['DiasABP', 'SysABP', 'MAP', 'NIDiasABP', 'NISysABP', 'NIMAP'],
    'Blood Gas': ['pH', 'PaO2', 'PaCO2', 'HCO3', 'SaO2', 'FiO2'],
    'Electrolytes': ['Na', 'K', 'Mg'],
    'Liver Function': ['ALT', 'AST', 'Bilirubin', 'ALP', 'Albumin'],
    'Kidney Function': ['BUN', 'Creatinine', 'Urine'],
    'Cardiac': ['TroponinT', 'TroponinI', 'HR'],
    'Hematology': ['WBC', 'HCT', 'Platelets'],
    'Metabolic': ['Glucose', 'Lactate', 'Cholesterol'],
    'Other': ['Temp', 'RespRate', 'GCS']
}

N_FEATURES = 35

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})

# =============================================================================
# Helper Functions
# =============================================================================

def load_feature_embedding(model_path):
    """Load Feature Identity Embedding from trained HELIX model."""
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    
    embedding_key = None
    for key in state_dict.keys():
        if 'feature_id' in key or 'feature_embed' in key:
            embedding_key = key
            break
    
    if embedding_key is None:
        raise KeyError("Could not find feature embedding in model")
    
    embedding = state_dict[embedding_key].numpy()
    print(f"Loaded embedding with shape: {embedding.shape}")
    return embedding


def get_feature_group(feature_name):
    """Get the semantic group of a feature."""
    for group, features in FEATURE_GROUPS.items():
        if feature_name in features:
            return group
    return 'Other'


def compute_within_between_stats(embedding):
    """Compute within-group and between-group similarities."""
    sim_matrix = cosine_similarity(embedding)
    
    within_sims = []
    between_sims = []
    
    for i in range(N_FEATURES):
        for j in range(i + 1, N_FEATURES):
            group_i = get_feature_group(PHYSIONET_FEATURES[i])
            group_j = get_feature_group(PHYSIONET_FEATURES[j])
            sim = sim_matrix[i, j]
            
            if group_i == group_j:
                within_sims.append(sim)
            else:
                between_sims.append(sim)
    
    return within_sims, between_sims


# =============================================================================
# Visualization
# =============================================================================

def plot_figure6(embedding, output_path):
    """
    Create Figure 6: Boxplot comparing within-group vs between-group similarity.
    """
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    
    within_sims, between_sims = compute_within_between_stats(embedding)
    
    bp = ax.boxplot([within_sims, between_sims], positions=[1, 2], widths=0.5, patch_artist=True)
    
    colors = ['#2ecc71', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Within-Group', 'Between-Group'], fontsize=12)
    ax.set_ylabel('Cosine Similarity', fontsize=12)
    
    # 统计检验
    stat, p_val = mannwhitneyu(within_sims, between_sims, alternative='greater')
    within_mean = np.mean(within_sims)
    between_mean = np.mean(between_sims)

    ax.text(1.5, 0.5, f'Within: {within_mean:.3f}\nBetween: {between_mean:.3f}\np = {p_val:.4f}',
            fontsize=9, verticalalignment='top', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_title('Feature Embedding Similarity:\nWithin-Group vs Between-Group', fontsize=13)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.02)
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', pad_inches=0.02)
    print(f"Saved: {output_path}")
    plt.close()
    
    return within_mean, between_mean, p_val


def main():
    MODEL_PATH = "/home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/reproduce_imputation/point01_log/PhysioNet2012_log/HELIX_PhysioNet2012/round_0/20260113_T101847/HELIX.pypots"
    OUTPUT_DIR = "/home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/physionet_embedding_analysis"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("PhysioNet2012 Feature Embedding - Figure 6")
    print("=" * 60)
    
    # Load embedding
    embedding = load_feature_embedding(MODEL_PATH)
    
    # Generate figure
    print("\nGenerating Figure 6...")
    within_mean, between_mean, p_val = plot_figure6(
        embedding, os.path.join(OUTPUT_DIR, "figure6_physionet_boxplot.pdf"))
    
    print(f"\nResults:")
    print(f"  Within-group mean:  {within_mean:.3f}")
    print(f"  Between-group mean: {between_mean:.3f}")
    print(f"  Mann-Whitney p-value: {p_val:.6f}")
    print(f"  Difference: {within_mean - between_mean:.3f}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()