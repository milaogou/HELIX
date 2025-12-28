#!/bin/bash
#SBATCH -x paraai-n32-h-01-agent-[1,4,8,16,17,25,27,28,29,30,31]
#SBATCH --job-name=NonstationaryTransformer_PeMS_point05
#SBATCH -o reproduce_imputation/point05_log/PeMS_log/NonstationaryTransformer_PeMS.out
#SBATCH -e reproduce_imputation/point05_log/PeMS_log/NonstationaryTransformer_PeMS.log
module purge
module load miniforge3/24.1 
module load compilers/cuda/12.1   compilers/gcc/11.3.0   cudnn/8.8.1.3_cuda12.x
source activate py310pots
export PYTHONUNBUFFERED=1
export http_proxy=http://u-cEoRwn:EDvFuZTe@172.16.4.9:3128
export https_proxy=http://u-cEoRwn:EDvFuZTe@172.16.4.9:3128   
python -u train_model.py --model NonstationaryTransformer --dataset PeMS --dataset_fold_path data/generated_datasets/pems_traffic_rate05_step24_point --saving_path reproduce_imputation/point05_log/PeMS_log --device cuda:0
