#!/bin/bash
#SBATCH -x paraai-n32-h-01-agent-[1,4,8,16,17,25,27,28,29,30,31]
#SBATCH --job-name=ImputeFormer_ETT_h1_point05
#SBATCH -o reproduce_imputation/point05_log/ETT_h1_log/ImputeFormer_ETT_h1.out
#SBATCH -e reproduce_imputation/point05_log/ETT_h1_log/ImputeFormer_ETT_h1.log
module purge
module load miniforge3/24.1 
module load compilers/cuda/12.1   compilers/gcc/11.3.0   cudnn/8.8.1.3_cuda12.x
source activate py310pots
export PYTHONUNBUFFERED=1
export http_proxy=http://u-cEoRwn:EDvFuZTe@172.16.4.9:3128
export https_proxy=http://u-cEoRwn:EDvFuZTe@172.16.4.9:3128   
python -u train_model.py --model ImputeFormer --dataset ETT_h1 --dataset_fold_path data/generated_datasets/ett_rate05_step48_point --saving_path reproduce_imputation/point05_log/ETT_h1_log --device cuda:0
