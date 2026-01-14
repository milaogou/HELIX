#!/bin/bash
#SBATCH -x paraai-n32-h-01-agent-[1,4,8,16,17,25,27,28,29,30,31]
#SBATCH --job-name=figure5_vis
#SBATCH -o /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/imputation_visualization/figure5.out
#SBATCH -e /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/imputation_visualization/figure5.log

# This is a CPU-only task (data loading + matplotlib plotting)
# No GPU needed

module purge
module load miniforge3/24.1 
module load compilers/cuda/12.1 compilers/gcc/11.3.0 cudnn/8.8.1.3_cuda12.x
source activate py310pots

export PYTHONUNBUFFERED=1
export http_proxy=http://u-cEoRwn:EDvFuZTe@172.16.4.9:3128
export https_proxy=http://u-cEoRwn:EDvFuZTe@172.16.4.9:3128
export LD_PRELOAD=$LD_PRELOAD:/home/bingxing2/home/scx7644/.conda/envs/py310pots/lib/python3.10/site-packages/sklearn/utils/../../scikit_learn.libs/libgomp-947d5fa1.so.1.0.0

# Create output directory
mkdir -p /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/imputation_visualization

cd /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code

echo "=============================================="
echo "Starting Figure 5 Generation"
echo "Time: $(date)"
echo "=============================================="

python -u imputation_visualization.py \
    --output_dir /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/imputation_visualization

echo "=============================================="
echo "Finished at: $(date)"
echo "=============================================="