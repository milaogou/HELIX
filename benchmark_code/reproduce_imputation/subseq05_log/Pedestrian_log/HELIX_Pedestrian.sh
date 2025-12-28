#!/bin/bash
#SBATCH -x paraai-n32-h-01-agent-[1,4,8,16,17,25,27,28,29,30,31]
#SBATCH --job-name=HELIX_Pedestrian_subseq05
#SBATCH -o reproduce_imputation/subseq05_log/Pedestrian_log/HELIX_Pedestrian.out
#SBATCH -e reproduce_imputation/subseq05_log/Pedestrian_log/HELIX_Pedestrian.log
module purge
module load miniforge3/24.1 
module load compilers/cuda/12.1   compilers/gcc/11.3.0   cudnn/8.8.1.3_cuda12.x
source activate py310pots
export PYTHONUNBUFFERED=1   
python -u train_model.py --model HELIX --dataset Pedestrian --dataset_fold_path data/generated_datasets/melbourne_pedestrian_rate05_step24_subseq_seqlen18 --saving_path reproduce_imputation/subseq05_log/Pedestrian_log --device cuda:0
