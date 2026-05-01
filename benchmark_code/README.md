# Experiment Code for HELIX

This directory contains all code to reproduce the experiments in the HELIX paper (ICML 2026).

## Prerequisites

- Linux aarch64 cluster with SLURM and NVIDIA A100 GPUs
- Conda environment: `conda env create -f conda_env.yml`
- PyPOTS (latest): `pip install pypots`

## Workflow

### 1. Dataset Generation

```bash
cd data/
python dataset_generating_point01.py
python dataset_generating_point05.py
python dataset_generating_point09.py
python dataset_generating_block05.py
python dataset_generating_subseq05.py
# Verify missing rates
python validate_missing_rate.py
```

Generated datasets are saved to `data/generated_datasets/` (gitignored due to size).

### 2. Hyperparameter Search (Optional)

Pre-tuned hyperparameters are already provided in `hpo_results/`. To re-run HPO:

```bash
cd PyPOTS_tuning_configs/
# Each model subfolder contains tuning configs
# Submit HPO jobs via:
python batch_hyperparameter_tuning.py
# Analyze results:
python analyze_tuning_results.py
# Apply best configs to hpo_results/:
python apply_tuned_configs.py
```

The tuning script calls `train_model_tuning.py` (different from the main experiment script).

### 3. Main Experiments

`in_sample_exp.py` reads hyperparameters from `hpo_results/` and submits SLURM jobs:

```bash
python in_sample_exp.py
```

Each job calls `train_model.py` which trains a model with 5 random seeds and reports mean±std. Adjust the `MODELS` and `dataset_folders` lists in `in_sample_exp.py` to select which experiments to run.

### 4. Result Collection & Analysis

```bash
cd reproduce_imputation/
# Parse training logs into CSV tables
python merge_naive_and_rank.py
# Statistical significance tests
python analyze_significance.py
# Advanced analysis (by-pattern, vs-naive, win-rate)
python analyze_advanced.py
```

Results are saved to `reproduce_imputation/results_csv/`.

### 5. LaTeX Table & Figure Generation

```bash
# Main paper and appendix tables (MAE)
python generate_latex_tables.py
# Multi-metric ranking table (MAE/MSE/MRE)
python generate_multi_metric_table.py
# MSE appendix tables
python generate_mse_appendix.py
```

### 6. Visualization & Analysis Scripts

| Script | Description |
|--------|-------------|
| `feature_embedding_analysis.py` | BeijingAir embedding structure (Fig. 2) |
| `extract_attention.py` | Attention pattern extraction (Fig. 3, 4) |
| `imputation_visualization.py` | Imputation quality visualization (Fig. 5) |
| `physionet_embedding_analysis.py` | PhysioNet2012 clinical grouping analysis (Fig. 6) |

## Directory Structure

```
benchmark_code/
├── train_model.py                 # Main training script (5 seeds)
├── train_model_tuning.py          # HPO training script
├── in_sample_exp.py               # Batch SLURM job submission
├── global_config.py               # Global configuration
├── utils.py                       # Utilities
├── data/                          # Dataset generation scripts
├── hpo_results/                   # Selected hyperparameters per dataset
├── PyPOTS_tuning_configs/         # HPO search spaces and scripts
├── reproduce_imputation/
│   ├── results_csv/               # Collected results (MAE/MSE/MRE)
│   ├── merge_naive_and_rank.py
│   ├── analyze_significance.py
│   └── analyze_advanced.py
├── generate_latex_tables.py       # LaTeX table generation
├── generate_multi_metric_table.py
├── generate_mse_appendix.py
├── latex_tables/                  # Generated .tex files
└── [analysis scripts]             # Figure generation scripts
```