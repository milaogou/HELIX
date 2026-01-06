#!/bin/bash
#SBATCH -x paraai-n32-h-01-agent-[1,4,8,16,17,25,27,28,29,30,31]
#SBATCH --job-name=TimeMixer_Electricity_point05
#SBATCH -o reproduce_imputation/point05_log/Electricity_log/TimeMixer_Electricity.out
#SBATCH -e reproduce_imputation/point05_log/Electricity_log/TimeMixer_Electricity.log
module purge
module load miniforge3/24.1 
module load compilers/cuda/12.1   compilers/gcc/11.3.0   cudnn/8.8.1.3_cuda12.x
source activate py310pots
export PYTHONUNBUFFERED=1
export http_proxy=http://u-cEoRwn:EDvFuZTe@172.16.4.9:3128
export https_proxy=http://u-cEoRwn:EDvFuZTe@172.16.4.9:3128   
export LD_PRELOAD=$LD_PRELOAD:/home/bingxing2/home/scx7644/.conda/envs/py310pots/lib/python3.10/site-packages/sklearn/utils/../../scikit_learn.libs/libgomp-947d5fa1.so.1.0.0
python -u train_model.py --model TimeMixer --dataset Electricity --dataset_fold_path data/generated_datasets/electricity_load_diagrams_rate05_step96_point --saving_path reproduce_imputation/point05_log/Electricity_log --device cuda:0
