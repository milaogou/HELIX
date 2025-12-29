#!/bin/bash
#SBATCH -x paraai-n32-h-01-agent-[1,4,8,16,17,25,27,28,29,30,31]
#SBATCH --job-name=tune_ModernTCN_ETT_h1_t1
#SBATCH -o /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/hyperparameter_tuning_results/ModernTCN_ETT_h1_tuning/trial_1.out
#SBATCH -e /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/hyperparameter_tuning_results/ModernTCN_ETT_h1_tuning/trial_1.err
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
cat > /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/hyperparameter_tuning_results/ModernTCN_ETT_h1_tuning/trial_1_params.json <<'EOF'
{
  "n_steps": 48,
  "n_features": 7,
  "epochs": 1000,
  "patience": 10,
  "patch_size": 6,
  "patch_stride": 6,
  "downsampling_ratio": 4,
  "ffn_ratio": 4,
  "num_blocks": [
    1,
    1
  ],
  "large_size": [
    7,
    7
  ],
  "small_size": [
    3,
    3
  ],
  "dims": [
    32,
    64
  ],
  "small_kernel_merged": false,
  "backbone_dropout": 0,
  "head_dropout": 0.1,
  "use_multi_scale": false,
  "individual": false,
  "apply_nonstationary_norm": false,
  "batch_size": 32,
  "lr": 0.006712129634578652
}
EOF

# 运行训练 - 使用新的 train_model_tuning.py
python -u train_model_tuning.py \
    --model ModernTCN \
    --dataset ETT_h1 \
    --dataset_fold_path data/generated_datasets/ett_rate01_step48_point \
    --saving_path /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/hyperparameter_tuning_results/ModernTCN_ETT_h1_tuning/trial_1 \
    --device cuda:0 \
    --n_rounds 1 \
    --n_steps 48 --n_features 7 --epochs 1000 --patience 10 --patch_size 6 --patch_stride 6 --downsampling_ratio 4 --ffn_ratio 4 --num_blocks [1,1] --large_size [7,7] --small_size [3,3] --dims [32,64] --small_kernel_merged False --backbone_dropout 0 --head_dropout 0.100000 --use_multi_scale False --individual False --apply_nonstationary_norm False --batch_size 32 --lr 0.006712

# 标记完成
echo "Trial 1 completed at $(date)" >> /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/hyperparameter_tuning_results/ModernTCN_ETT_h1_tuning/trial_1_status.txt
