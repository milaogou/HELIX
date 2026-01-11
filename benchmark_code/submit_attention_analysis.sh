#!/bin/bash
#SBATCH -x paraai-n32-h-01-agent-[1,4,8,16,17,25,27,28,29,30,31]
#SBATCH --job-name=HELIX_attn_analysis
#SBATCH -o /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/attention_analysis/attention_analysis.out
#SBATCH -e /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/attention_analysis/attention_analysis.log
#SBATCH --gpus=1

module purge
module load miniforge3/24.1 
module load compilers/cuda/12.1 compilers/gcc/11.3.0 cudnn/8.8.1.3_cuda12.x
source activate py310pots

export PYTHONUNBUFFERED=1
export http_proxy=http://u-cEoRwn:EDvFuZTe@172.16.4.9:3128
export https_proxy=http://u-cEoRwn:EDvFuZTe@172.16.4.9:3128
export LD_PRELOAD=$LD_PRELOAD:/home/bingxing2/home/scx7644/.conda/envs/py310pots/lib/python3.10/site-packages/sklearn/utils/../../scikit_learn.libs/libgomp-947d5fa1.so.1.0.0

cd /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code

python -u extract_attention.py \
    --model_path /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/reproduce_imputation/point01_log/BeijingAir_log/HELIX_BeijingAir/round_0/20260110_T174206/HELIX.pypots \
    --data_path /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/data/generated_datasets/beijing_air_quality_rate01_step24_point/test.h5 \
    --output_dir /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/attention_analysis \
    --device cuda:0 \
    --n_samples 4