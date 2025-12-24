#!/bin/bash
#SBATCH -x paraai-n32-h-01-agent-[1,4,8,16,17,25,27,28,29,30,31]
#SBATCH --job-name=HELIX_Pedestrian_point09
#SBATCH -o reproduce_imputation/point09_log/Pedestrian_log/HELIX_Pedestrian.out
#SBATCH -e reproduce_imputation/point09_log/Pedestrian_log/HELIX_Pedestrian.log
module purge
module load miniforge3/24.1 
module load compilers/cuda/12.1   compilers/gcc/11.3.0   cudnn/8.8.1.3_cuda12.x
source activate py310pots
export PYTHONUNBUFFERED=1   
python -u train_model.py --model HELIX --dataset Pedestrian --dataset_fold_path data/generated_datasets/melbourne_pedestrian_rate09_step24_point --saving_path reproduce_imputation/point09_log/Pedestrian_log --device cuda:0
