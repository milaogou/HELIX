#!/bin/bash
#SBATCH -x paraai-n32-h-01-agent-[1,4,8,16,17,25,27,28,29,30,31]
#SBATCH --job-name=tune_TimeMixerPP_ETT_h1_t19
#SBATCH -o /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/hyperparameter_tuning_results/TimeMixerPP_ETT_h1_tuning/trial_19.out
#SBATCH -e /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/hyperparameter_tuning_results/TimeMixerPP_ETT_h1_tuning/trial_19.err
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
cat > /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/hyperparameter_tuning_results/TimeMixerPP_ETT_h1_tuning/trial_19_params.json <<'EOF'
{
  "n_steps": 48,
  "n_features": 7,
  "epochs": 1000,
  "patience": 10,
  "n_layers": 3,
  "d_model": 64,
  "d_ffn": 256,
  "top_k": 5,
  "n_heads": 2,
  "n_kernels": 8,
  "dropout": 0.2,
  "channel_mixing": true,
  "channel_independence": false,
  "downsampling_layers": 2,
  "downsampling_window": 2,
  "apply_nonstationary_norm": false,
  "batch_size": 16,
  "lr": 0.00022936431257838702
}
EOF

# 运行训练
python -u train_model_tuning.py \
    --model TimeMixerPP \
    --dataset ETT_h1 \
    --dataset_fold_path data/generated_datasets/ett_rate01_step48_point \
    --saving_path /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/hyperparameter_tuning_results/TimeMixerPP_ETT_h1_tuning/trial_19 \
    --device cuda:0 \
    --n_rounds 1 \
    --n_steps 48 --n_features 7 --epochs 1000 --patience 10 --n_layers 3 --d_model 64 --d_ffn 256 --top_k 5 --n_heads 2 --n_kernels 8 --dropout 0.200000 --channel_mixing True --channel_independence False --downsampling_layers 2 --downsampling_window 2 --apply_nonstationary_norm False --batch_size 16 --lr 0.000229

# 标记完成
echo "Trial 19 completed at $(date)" >> /home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/hyperparameter_tuning_results/TimeMixerPP_ETT_h1_tuning/trial_19_status.txt
