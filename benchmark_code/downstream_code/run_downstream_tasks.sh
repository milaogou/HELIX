#!/bin/bash
#SBATCH -x paraai-n32-h-01-agent-[1,4,8,16,17,25,27,28,29,30,31]
module purge
module load miniforge3/24.1 
module load compilers/cuda/12.1   compilers/gcc/11.3.0   cudnn/8.8.1.3_cuda12.x
source activate py310pots
export PYTHONUNBUFFERED=1
export http_proxy=http://u-cEoRwn:EDvFuZTe@172.16.4.9:3128
export https_proxy=http://u-cEoRwn:EDvFuZTe@172.16.4.9:3128   
export LD_PRELOAD=$LD_PRELOAD:/home/bingxing2/home/scx7644/.conda/envs/py310pots/lib/python3.10/site-packages/sklearn/utils/../../scikit_learn.libs/libgomp-947d5fa1.so.1.0.0
export LD_PRELOAD=$LD_PRELOAD:/home/bingxing2/home/scx7644/.conda/envs/py310pots/lib/python3.10/site-packages/xgboost/lib/../../xgboost.libs/libgomp-d22c30c5.so.1.0.0

# Paths
BASE_DIR="/home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code"
DATA_DIR="${BASE_DIR}/data/generated_datasets"
IMPUTATION_DIR="${BASE_DIR}/reproduce_imputation"
OUTPUT_DIR="${BASE_DIR}/downstream_result"

# Datasets
PHYSIONET_PATH="${DATA_DIR}/physionet_2012_rate01_point"
ETT_PATH="${DATA_DIR}/ett_rate03_step48_block_blocklen6"

# Model directories
PHYSIONET_LOG_DIR="${IMPUTATION_DIR}/point01_log/PhysioNet2012_log"
ETT_LOG_DIR="${IMPUTATION_DIR}/block03_log/ETT_h1_log"

# Create output directories
mkdir -p ${OUTPUT_DIR}/classification/logs
mkdir -p ${OUTPUT_DIR}/forecasting/logs
mkdir -p ${OUTPUT_DIR}/regression/logs

# 18 DL Models
DL_MODELS=(
    "HELIX"
    # "HELIX_NoFeatureEmbed"
    # "HELIX_NoFusion"
    # "HELIX_NoHybrid"
    # "HELIX_NoRotaryPE"
    # "TEFN"
    # "TimeMixerPP"
    # "TimeLLM"
    # "MOMENT"
    # "TimeMixer"
    # "ModernTCN"
    # "ImputeFormer"
    # "TOTEM"
    # "iTransformer"
    # "SAITS"
    # "FreTS"
    # "NonstationaryTransformer"
    # "PatchTST"
)

# 4 Naive Methods
NAIVE_METHODS=("mean" "median" "locf" "linear_interpolation")

echo "=========================================="
echo "Starting Downstream Tasks"
echo "=========================================="

# ==========================================
# Classification Task (PhysioNet2012)
# ==========================================
echo ""
echo "=== Classification Task (PhysioNet2012) ==="

# Run DL models
for MODEL in "${DL_MODELS[@]}"; do
    MODEL_PATH="${PHYSIONET_LOG_DIR}/${MODEL}_PhysioNet2012"
    LOG_FILE="${OUTPUT_DIR}/classification/logs/${MODEL}.log"
    
    if [ -d "$MODEL_PATH" ]; then
        echo "Running classification for ${MODEL}..."
        python downstream_code/downstream_classification.py \
            --device cuda:0 \
            --model ${MODEL} \
            --dataset PhysioNet2012 \
            --dataset_fold_path ${PHYSIONET_PATH} \
            --n_classes 2 \
            --model_result_parent_fold ${MODEL_PATH} \
            > ${LOG_FILE} 2>&1
        echo "  Done. Log saved to ${LOG_FILE}"
    else
        echo "  Warning: ${MODEL_PATH} not found, skipping..."
    fi
done

# Run Naive methods
for METHOD in "${NAIVE_METHODS[@]}"; do
    LOG_FILE="${OUTPUT_DIR}/classification/logs/Naive_${METHOD}.log"
    echo "Running classification for Naive_${METHOD}..."
    python downstream_code/downstream_classification_naive.py \
        --dataset_fold_path ${PHYSIONET_PATH} \
        --naive_method ${METHOD} \
        --n_classes 2 \
        > ${LOG_FILE} 2>&1
    echo "  Done. Log saved to ${LOG_FILE}"
done

# ==========================================
# Forecasting Task (ETT_h1)
# ==========================================
echo ""
echo "=== Forecasting Task (ETT_h1) ==="

# Run DL models
for MODEL in "${DL_MODELS[@]}"; do
    MODEL_PATH="${ETT_LOG_DIR}/${MODEL}_ETT_h1"
    LOG_FILE="${OUTPUT_DIR}/forecasting/logs/${MODEL}.log"
    
    if [ -d "$MODEL_PATH" ]; then
        echo "Running forecasting for ${MODEL}..."
        python downstream_code/downstream_forecasting.py \
            --device cuda:0 \
            --model ${MODEL} \
            --dataset ETT_h1 \
            --dataset_fold_path ${ETT_PATH} \
            --model_result_parent_fold ${MODEL_PATH} \
            > ${LOG_FILE} 2>&1
        echo "  Done. Log saved to ${LOG_FILE}"
    else
        echo "  Warning: ${MODEL_PATH} not found, skipping..."
    fi
done

# Run Naive methods
for METHOD in "${NAIVE_METHODS[@]}"; do
    LOG_FILE="${OUTPUT_DIR}/forecasting/logs/Naive_${METHOD}.log"
    echo "Running forecasting for Naive_${METHOD}..."
    python downstream_code/downstream_forecasting_naive.py \
        --dataset_fold_path ${ETT_PATH} \
        --naive_method ${METHOD} \
        > ${LOG_FILE} 2>&1
    echo "  Done. Log saved to ${LOG_FILE}"
done

# ==========================================
# Regression Task (ETT_h1)
# ==========================================
echo ""
echo "=== Regression Task (ETT_h1) ==="

# Run DL models
for MODEL in "${DL_MODELS[@]}"; do
    MODEL_PATH="${ETT_LOG_DIR}/${MODEL}_ETT_h1"
    LOG_FILE="${OUTPUT_DIR}/regression/logs/${MODEL}.log"
    
    if [ -d "$MODEL_PATH" ]; then
        echo "Running regression for ${MODEL}..."
        python downstream_code/downstream_regression.py \
            --device cuda:0 \
            --model ${MODEL} \
            --dataset ETT_h1 \
            --dataset_fold_path ${ETT_PATH} \
            --model_result_parent_fold ${MODEL_PATH} \
            > ${LOG_FILE} 2>&1
        echo "  Done. Log saved to ${LOG_FILE}"
    else
        echo "  Warning: ${MODEL_PATH} not found, skipping..."
    fi
done

# Run Naive methods
for METHOD in "${NAIVE_METHODS[@]}"; do
    LOG_FILE="${OUTPUT_DIR}/regression/logs/Naive_${METHOD}.log"
    echo "Running regression for Naive_${METHOD}..."
    python downstream_code/downstream_regression_naive.py \
        --dataset_fold_path ${ETT_PATH} \
        --naive_method ${METHOD} \
        > ${LOG_FILE} 2>&1
    echo "  Done. Log saved to ${LOG_FILE}"
done

echo ""
echo "=========================================="
echo "All downstream tasks completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=========================================="

# Run result parsing
echo ""
echo "Parsing results..."
python downstream_code/parse_downstream_results.py --output_dir ${OUTPUT_DIR}
echo "Done!"