#!/bin/bash
#SBATCH -x paraai-n32-h-01-agent-[1,4,8,16,17,25,27,28,29,30,31]
#SBATCH --job-name=HELIX_BeijingAir
###SBATCH --output=/home/bingxing2/home/scx7644/MASTER/Awesome_Imputation-main/benchmark_code/reproduce_imputation/point01_log/BeijingAir_log/HELIX_BeijingAir.out
###SBATCH --error=/home/bingxing2/home/scx7644/MASTER/Awesome_Imputation-main/benchmark_code/reproduce_imputation/point01_log/BeijingAir_log/HELIX_BeijingAir.log
module purge
module load miniforge3/24.1 compilers/cuda/12.1 cudnn/8.8.1.3_cuda12.x compilers/gcc/9.3.0 
source activate /home/bingxing2/home/scx7644/.conda/envs/py39_env
export LD_LIBRARY_PATH=/home/bingxing2/apps/cuda/11.6.0/targets/sbsa-linux/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1   
python -u train_model.py --model HELIX --dataset BeijingAir --dataset_fold_path data/generated_datasets/beijing_air_quality_rate01_step24_point --saving_path reproduce_imputation/point01_log/BeijingAir_log --device cuda:0


