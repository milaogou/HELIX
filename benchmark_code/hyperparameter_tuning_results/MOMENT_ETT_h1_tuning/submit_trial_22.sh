#!/bin/bash
#SBATCH -x paraai-n32-h-01-agent-[1,4,8,16,17,25,27,28,29,30,31]
#SBATCH --job-name=tune_MOMENT_ETT_h1_t22
#SBATCH -o /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/hyperparameter_tuning_results/MOMENT_ETT_h1_tuning/trial_22.out
#SBATCH -e /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/hyperparameter_tuning_results/MOMENT_ETT_h1_tuning/trial_22.err
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
cat > /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/hyperparameter_tuning_results/MOMENT_ETT_h1_tuning/trial_22_params.json <<'EOF'
{
  "n_steps": 48,
  "n_features": 7,
  "epochs": 1000,
  "patience": 10,
  "patch_size": 12,
  "patch_stride": 12,
  "transformer_backbone": "t5-base",
  "transformer_type": "encoder_only",
  "d_model": 768,
  "revin_affine": true,
  "add_positional_embedding": true,
  "value_embedding_bias": true,
  "orth_gain": 1.41,
  "n_layers": 6,
  "d_ffn": 2048,
  "dropout": 0.1,
  "head_dropout": 0.2,
  "finetuning_mode": "linear-probing",
  "mask_ratio": 0.5,
  "batch_size": 16,
  "lr": 0.0003461856416604145
}
EOF

# 运行训练
python -u train_model_tuning.py \
    --model MOMENT \
    --dataset ETT_h1 \
    --dataset_fold_path data/generated_datasets/ett_rate01_step48_point \
    --saving_path /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/hyperparameter_tuning_results/MOMENT_ETT_h1_tuning/trial_22 \
    --device cuda:0 \
    --n_rounds 1 \
    --n_steps 48 --n_features 7 --epochs 1000 --patience 10 --patch_size 12 --patch_stride 12 --transformer_backbone t5-base --transformer_type encoder_only --d_model 768 --revin_affine True --add_positional_embedding True --value_embedding_bias True --orth_gain 1.410000 --n_layers 6 --d_ffn 2048 --dropout 0.100000 --head_dropout 0.200000 --finetuning_mode linear-probing --mask_ratio 0.500000 --batch_size 16 --lr 0.000346

# 标记完成
echo "Trial 22 completed at $(date)" >> /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/hyperparameter_tuning_results/MOMENT_ETT_h1_tuning/trial_22_status.txt
