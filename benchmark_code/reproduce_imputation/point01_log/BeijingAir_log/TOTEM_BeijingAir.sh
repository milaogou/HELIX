#!/bin/bash
#SBATCH -x paraai-n32-h-01-agent-[1,4,8,16,17,25,27,28,29,30,31]
#SBATCH --job-name=TOTEM_BeijingAir_point01
#SBATCH -o reproduce_imputation/point01_log/BeijingAir_log/TOTEM_BeijingAir.out
#SBATCH -e reproduce_imputation/point01_log/BeijingAir_log/TOTEM_BeijingAir.log
module purge
module load miniforge3/24.1 
module load compilers/cuda/12.1   compilers/gcc/11.3.0   cudnn/8.8.1.3_cuda12.x
source activate py310pots
export PYTHONUNBUFFERED=1   
python -u train_model.py --model TOTEM --dataset BeijingAir --dataset_fold_path data/generated_datasets/beijing_air_quality_rate01_step24_point --saving_path reproduce_imputation/point01_log/BeijingAir_log --device cuda:0
