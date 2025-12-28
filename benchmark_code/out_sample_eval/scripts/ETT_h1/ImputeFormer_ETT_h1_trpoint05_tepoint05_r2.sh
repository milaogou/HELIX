#!/bin/bash
#SBATCH -x paraai-n32-h-01-agent-[1,4,8,16,17,25,27,28,29,30,31]
#SBATCH --job-name=ImputeFormer_ETT_h1_trpoint05_tepoint05_r2
#SBATCH -o out_sample_eval/logs/ETT_h1/ImputeFormer_ETT_h1_trpoint05_tepoint05_r2.out
#SBATCH -e out_sample_eval/logs/ETT_h1/ImputeFormer_ETT_h1_trpoint05_tepoint05_r2.log
#SBATCH --gpus=1

module purge
module load miniforge3/24.1 
module load compilers/cuda/12.1 compilers/gcc/11.3.0 cudnn/8.8.1.3_cuda12.x
source activate py310pots

export PYTHONUNBUFFERED=1
export http_proxy=http://u-cEoRwn:EDvFuZTe@172.16.4.9:3128
export https_proxy=http://u-cEoRwn:EDvFuZTe@172.16.4.9:3128
export LD_PRELOAD=$LD_PRELOAD:/home/bingxing2/home/scx7644/.conda/envs/py310pots/lib/python3.10/site-packages/sklearn/utils/../../scikit_learn.libs/libgomp-947d5fa1.so.1.0.0
python -u out_sample_eval.py \
    --model ImputeFormer \
    --dataset ETT_h1 \
    --train_pattern point05 \
    --test_pattern ett_rate05_step48_point \
    --round_id 2 \
    --model_base_path reproduce_imputation \
    --data_base_path data/generated_datasets \
    --output_base_path out_sample_eval \
    --device cuda:0
