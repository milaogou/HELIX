#!/bin/bash
#SBATCH -x paraai-n32-h-01-agent-[1,4,8,16,17,25,27,28,29,30,31]
#SBATCH --job-name=tune_StemGNN_BeijingAir_t25
#SBATCH -o /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/hyperparameter_tuning_results/StemGNN_BeijingAir_tuning/trial_25.out
#SBATCH -e /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/hyperparameter_tuning_results/StemGNN_BeijingAir_tuning/trial_25.err
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
cat > /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/hyperparameter_tuning_results/StemGNN_BeijingAir_tuning/trial_25_params.json <<'EOF'
{
  "n_steps": 24,
  "n_features": 132,
  "epochs": 1000,
  "patience": 10,
  "n_layers": 2,
  "n_stacks": 1,
  "d_model": 64,
  "dropout": 0.5,
  "leaky_rate": 0.3,
  "ORT_weight": 1.0,
  "MIT_weight": 1.0,
  "batch_size": 8,
  "lr": 0.0001582790356197523
}
EOF

# 运行训练
python -u train_model_tuning.py \
    --model StemGNN \
    --dataset BeijingAir \
    --dataset_fold_path data/generated_datasets/beijing_air_quality_rate01_step24_point \
    --saving_path /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/hyperparameter_tuning_results/StemGNN_BeijingAir_tuning/trial_25 \
    --device cuda:0 \
    --n_rounds 1 \
    --n_steps 24 --n_features 132 --epochs 1000 --patience 10 --n_layers 2 --n_stacks 1 --d_model 64 --dropout 0.500000 --leaky_rate 0.300000 --ORT_weight 1.000000 --MIT_weight 1.000000 --batch_size 8 --lr 0.000158

# 标记完成
echo "Trial 25 completed at $(date)" >> /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/hyperparameter_tuning_results/StemGNN_BeijingAir_tuning/trial_25_status.txt
