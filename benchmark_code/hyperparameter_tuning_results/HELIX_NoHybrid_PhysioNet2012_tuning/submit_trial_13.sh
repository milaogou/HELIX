#!/bin/bash
#SBATCH -x paraai-n32-h-01-agent-[1,4,8,16,17,25,27,28,29,30,31]
#SBATCH --job-name=tune_HELIX_NoHybrid_PhysioNet2012_t13
#SBATCH -o /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/hyperparameter_tuning_results/HELIX_NoHybrid_PhysioNet2012_tuning/trial_13.out
#SBATCH -e /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/hyperparameter_tuning_results/HELIX_NoHybrid_PhysioNet2012_tuning/trial_13.err
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
cat > /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/hyperparameter_tuning_results/HELIX_NoHybrid_PhysioNet2012_tuning/trial_13_params.json <<'EOF'
{
  "n_steps": 48,
  "n_features": 35,
  "epochs": 1000,
  "patience": 10,
  "pe_dim": 12,
  "feature_embed_dim": 16,
  "d_model": 128,
  "n_heads": 4,
  "n_layers": 1,
  "dropout": 0,
  "ORT_weight": 1.0,
  "MIT_weight": 1.0,
  "batch_size": 32,
  "lr": 0.00011608216139263449
}
EOF

# 运行训练
python -u train_model_tuning.py \
    --model HELIX_NoHybrid \
    --dataset PhysioNet2012 \
    --dataset_fold_path data/generated_datasets/physionet_2012_rate01_point \
    --saving_path /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/hyperparameter_tuning_results/HELIX_NoHybrid_PhysioNet2012_tuning/trial_13 \
    --device cuda:0 \
    --n_rounds 1 \
    --n_steps 48 --n_features 35 --epochs 1000 --patience 10 --pe_dim 12 --feature_embed_dim 16 --d_model 128 --n_heads 4 --n_layers 1 --dropout 0 --ORT_weight 1.000000 --MIT_weight 1.000000 --batch_size 32 --lr 0.000116

# 标记完成
echo "Trial 13 completed at $(date)" >> /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/hyperparameter_tuning_results/HELIX_NoHybrid_PhysioNet2012_tuning/trial_13_status.txt
