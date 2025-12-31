#!/bin/bash
#SBATCH -x paraai-n32-h-01-agent-[1,4,8,16,17,25,27,28,29,30,31]
#SBATCH --job-name=tune_TOTEM_ItalyAir_t22
#SBATCH -o /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/hyperparameter_tuning_results/TOTEM_ItalyAir_tuning/trial_22.out
#SBATCH -e /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/hyperparameter_tuning_results/TOTEM_ItalyAir_tuning/trial_22.err
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
cat > /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/hyperparameter_tuning_results/TOTEM_ItalyAir_tuning/trial_22_params.json <<'EOF'
{
  "n_steps": 12,
  "n_features": 13,
  "epochs": 1000,
  "patience": 10,
  "d_block_hidden": 24,
  "n_residual_layers": 1,
  "d_residual_hidden": 8,
  "d_embedding": 16,
  "n_embeddings": 256,
  "commitment_cost": 0.5,
  "compression_factor": 4,
  "batch_size": 16,
  "lr": 0.0008056245437530991
}
EOF

# 运行训练
python -u train_model_tuning.py \
    --model TOTEM \
    --dataset ItalyAir \
    --dataset_fold_path data/generated_datasets/italy_air_quality_rate01_step12_point \
    --saving_path /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/hyperparameter_tuning_results/TOTEM_ItalyAir_tuning/trial_22 \
    --device cuda:0 \
    --n_rounds 1 \
    --n_steps 12 --n_features 13 --epochs 1000 --patience 10 --d_block_hidden 24 --n_residual_layers 1 --d_residual_hidden 8 --d_embedding 16 --n_embeddings 256 --commitment_cost 0.500000 --compression_factor 4 --batch_size 16 --lr 0.000806

# 标记完成
echo "Trial 22 completed at $(date)" >> /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/hyperparameter_tuning_results/TOTEM_ItalyAir_tuning/trial_22_status.txt
