#!/bin/bash
#SBATCH -x paraai-n32-h-01-agent-[1,4,8,16,17,25,27,28,29,30,31]
#SBATCH --job-name=tune_HELIX_ETT_h1_t13
#SBATCH -o /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/hyperparameter_tuning_results/HELIX_ETT_h1_tuning/trial_13.out
#SBATCH -e /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/hyperparameter_tuning_results/HELIX_ETT_h1_tuning/trial_13.err
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
cat > /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/hyperparameter_tuning_results/HELIX_ETT_h1_tuning/trial_13_params.json <<'EOF'
{
  "n_steps": 48,
  "n_features": 7,
  "epochs": 1000,
  "patience": 10,
  "pe_dim": 48,
  "feature_embed_dim": 24,
  "d_model": 96,
  "n_heads": 8,
  "n_layers": 2,
  "dropout": 0.3,
  "batch_size": 8,
  "lr": 0.00016739228996035313
}
EOF

# 运行训练
python -u train_model_tuning.py \
    --model HELIX \
    --dataset ETT_h1 \
    --dataset_fold_path data/generated_datasets/ett_rate05_step48_subseq_seqlen36 \
    --saving_path /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/hyperparameter_tuning_results/HELIX_ETT_h1_tuning/trial_13 \
    --device cuda:0 \
    --n_rounds 1 \
    --n_steps 48 --n_features 7 --epochs 1000 --patience 10 --pe_dim 48 --feature_embed_dim 24 --d_model 96 --n_heads 8 --n_layers 2 --dropout 0.300000 --batch_size 8 --lr 0.000167

# 标记完成
echo "Trial 13 completed at $(date)" >> /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/hyperparameter_tuning_results/HELIX_ETT_h1_tuning/trial_13_status.txt
