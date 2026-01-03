#!/bin/bash
#SBATCH -x paraai-n32-h-01-agent-[1,4,8,16,17,25,27,28,29,30,31]
#SBATCH --job-name=tune_TimeMixerPP_BeijingAir_t23
#SBATCH -o /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/hyperparameter_tuning_results/TimeMixerPP_BeijingAir_tuning/trial_23.out
#SBATCH -e /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/hyperparameter_tuning_results/TimeMixerPP_BeijingAir_tuning/trial_23.err
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
cat > /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/hyperparameter_tuning_results/TimeMixerPP_BeijingAir_tuning/trial_23_params.json <<'EOF'
{
  "n_steps": 24,
  "n_features": 132,
  "epochs": 1000,
  "patience": 10,
  "n_layers": 1,
  "d_model": 64,
  "d_ffn": 384,
  "top_k": 3,
  "n_heads": 4,
  "n_kernels": 6,
  "dropout": 0.1,
  "channel_mixing": true,
  "channel_independence": false,
  "downsampling_layers": 2,
  "downsampling_window": 4,
  "apply_nonstationary_norm": false,
  "batch_size": 16,
  "lr": 0.009168522907304374
}
EOF

# 运行训练
python -u train_model_tuning.py \
    --model TimeMixerPP \
    --dataset BeijingAir \
    --dataset_fold_path data/generated_datasets/beijing_air_quality_rate01_step24_point \
    --saving_path /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/hyperparameter_tuning_results/TimeMixerPP_BeijingAir_tuning/trial_23 \
    --device cuda:0 \
    --n_rounds 1 \
    --n_steps 24 --n_features 132 --epochs 1000 --patience 10 --n_layers 1 --d_model 64 --d_ffn 384 --top_k 3 --n_heads 4 --n_kernels 6 --dropout 0.100000 --channel_mixing True --channel_independence False --downsampling_layers 2 --downsampling_window 4 --apply_nonstationary_norm False --batch_size 16 --lr 0.009169

# 标记完成
echo "Trial 23 completed at $(date)" >> /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/hyperparameter_tuning_results/TimeMixerPP_BeijingAir_tuning/trial_23_status.txt
