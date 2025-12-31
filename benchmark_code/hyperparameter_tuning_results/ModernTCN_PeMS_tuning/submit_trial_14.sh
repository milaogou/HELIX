#!/bin/bash
#SBATCH -x paraai-n32-h-01-agent-[1,4,8,16,17,25,27,28,29,30,31]
#SBATCH --job-name=tune_ModernTCN_PeMS_t14
#SBATCH -o /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/hyperparameter_tuning_results/ModernTCN_PeMS_tuning/trial_14.out
#SBATCH -e /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/hyperparameter_tuning_results/ModernTCN_PeMS_tuning/trial_14.err
#SBATCH --gpus=1

module purge
module load miniforge3/24.1 
module load compilers/cuda/12.1 compilers/gcc/11.3.0 cudnn/8.8.1.3_cuda12.x
source activate py310pots

export PYTHONUNBUFFERED=1
export http_proxy=http://u-cEoRwn:EDvFuZTe@172.16.4.9:3128
export https_proxy=http://u-cEoRwn:EDvFuZTe@172.16.4.9:3128
export LD_PRELOAD=$LD_PRELOAD:/home/bingxing2/home/scx7644/.conda/envs/py310pots/lib/python3.10/site-packages/sklearn/utils/../../scikit_learn.libs/libgomp-947d5fa1.so.1.0.0
# 保存参数配置
cat > /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/hyperparameter_tuning_results/ModernTCN_PeMS_tuning/trial_14_params.json <<'EOF'
{
  "n_steps": 24,
  "n_features": 862,
  "epochs": 1000,
  "patience": 10,
  "patch_size": 8,
  "patch_stride": 6,
  "downsampling_ratio": 4,
  "ffn_ratio": 2,
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
    64,
    64
  ],
  "small_kernel_merged": false,
  "backbone_dropout": 0.2,
  "head_dropout": 0.2,
  "use_multi_scale": false,
  "individual": false,
  "apply_nonstationary_norm": false,
  "batch_size": 1,
  "lr": 0.0012129661873765741
}
EOF

# 运行训练
python -u train_model_tuning.py \
    --model ModernTCN \
    --dataset PeMS \
    --dataset_fold_path data/generated_datasets/pems_traffic_rate01_step24_point \
    --saving_path /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/hyperparameter_tuning_results/ModernTCN_PeMS_tuning/trial_14 \
    --device cuda:0 \
    --n_rounds 1 \
    --n_steps 24 --n_features 862 --epochs 1000 --patience 10 --patch_size 8 --patch_stride 6 --downsampling_ratio 4 --ffn_ratio 2 --num_blocks [1,1] --large_size [7,7] --small_size [3,3] --dims [64,64] --small_kernel_merged False --backbone_dropout 0.200000 --head_dropout 0.200000 --use_multi_scale False --individual False --apply_nonstationary_norm False --batch_size 1 --lr 0.001213

# 标记完成
echo "Trial 14 completed at $(date)" >> /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/hyperparameter_tuning_results/ModernTCN_PeMS_tuning/trial_14_status.txt
