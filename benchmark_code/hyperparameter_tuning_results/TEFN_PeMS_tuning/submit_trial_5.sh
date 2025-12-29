#!/bin/bash
#SBATCH -x paraai-n32-h-01-agent-[1,4,8,16,17,25,27,28,29,30,31]
#SBATCH --job-name=tune_TEFN_PeMS_t5
#SBATCH -o /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/hyperparameter_tuning_results/TEFN_PeMS_tuning/trial_5.out
#SBATCH -e /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/hyperparameter_tuning_results/TEFN_PeMS_tuning/trial_5.err
#SBATCH --gpus=1
#SBATCH --time=06:00:00

module purge
module load miniforge3/24.1 
module load compilers/cuda/12.1 compilers/gcc/11.3.0 cudnn/8.8.1.3_cuda12.x
source activate py310pots

export PYTHONUNBUFFERED=1
export http_proxy=http://u-cEoRwn:EDvFuZTe@172.16.4.9:3128
export https_proxy=http://u-cEoRwn:EDvFuZTe@172.16.4.9:3128
export LD_PRELOAD=$LD_PRELOAD:/home/bingxing2/home/scx7644/.conda/envs/py310pots/lib/python3.10/site-packages/sklearn/utils/../../scikit_learn.libs/libgomp-947d5fa1.so.1.0.0
# 保存参数配置
cat > /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/hyperparameter_tuning_results/TEFN_PeMS_tuning/trial_5_params.json <<'EOF'
{
  "n_steps": 24,
  "n_features": 862,
  "epochs": 1000,
  "patience": 10,
  "n_fod": 2,
  "apply_nonstationary_norm": false,
  "batch_size": 2,
  "lr": 0.0005232111979114645
}
EOF

# 运行训练 - 使用新的 train_model_tuning.py
python -u train_model_tuning.py \
    --model TEFN \
    --dataset PeMS \
    --dataset_fold_path data/generated_datasets/pems_traffic_rate01_step24_point \
    --saving_path /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/hyperparameter_tuning_results/TEFN_PeMS_tuning/trial_5 \
    --device cuda:0 \
    --n_rounds 1 \
    --n_steps 24 --n_features 862 --epochs 1000 --patience 10 --n_fod 2 --apply_nonstationary_norm False --batch_size 2 --lr 0.000523

# 标记完成
echo "Trial 5 completed at $(date)" >> /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/hyperparameter_tuning_results/TEFN_PeMS_tuning/trial_5_status.txt
