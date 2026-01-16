# -*- coding: utf-8 -*-
"""
批量超参数调优脚本 - 适配SLURM集群环境
"""
import os
import json
import time
import subprocess
import random
import math
from typing import Dict, List, Any
from pathlib import Path

# ==================== 配置参数 ====================
BASE_DIR = "/home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code"
DATA_BASE_PATH = "data/generated_datasets"
TUNING_OUTPUT_PATH = "hyperparameter_tuning_results"
CONFIG_BASE_PATH = "PyPOTS_tuning_configs"

# 调优配置 - 每个模型25个trials
TUNING_CONFIG = {
    'ETT_h1': {
        'models': ['StemGNN'],#'HELIX_NoRotaryPE','HELIX','HELIX_NoFeatureEmbed', 'HELIX_NoHybrid', 'HELIX_NoFusion','ModernTCN', 'TEFN', 'TimeMixer','MOMENT', 'ImputeFormer', ,'TOTEM','TimeMixerPP',   'TimeLLM', 
        'dataset_path': 'ett_rate01_step48_point',
        'max_trials_per_model': 25,
    },
    'PeMS': {
        'models': ['StemGNN'],#'HELIX_NoRotaryPE','HELIX', 'HELIX_NoFeatureEmbed', 'HELIX_NoHybrid', 'HELIX_NoFusion','ModernTCN', 'TEFN','TimeMixer','ImputeFormer','TOTEM','TimeMixerPP'
        'dataset_path': 'pems_traffic_rate01_step24_point',
        'max_trials_per_model': 25,
    },
    'BeijingAir': {
        'models': ['StemGNN'],#'HELIX_NoRotaryPE','HELIX', 'HELIX_NoFeatureEmbed', 'HELIX_NoHybrid', 'HELIX_NoFusion','ModernTCN','MOMENT','TOTEM','TEFN', 'TimeMixer','TimeMixerPP',   'ImputeFormer',  
        'dataset_path': 'beijing_air_quality_rate01_step24_point',
        'max_trials_per_model': 25,
    },
    'PhysioNet2012': {
        'models': ['StemGNN'],#'HELIX_NoRotaryPE','HELIX', 'HELIX_NoFeatureEmbed', 'HELIX_NoHybrid', 'HELIX_NoFusion','MOMENT','TOTEM', 'TEFN', 'TimeMixer', 'ModernTCN','TimeMixerPP', 'ImputeFormer',  
        'dataset_path': 'physionet_2012_rate01_point',
        'max_trials_per_model': 25,
    },
    'ItalyAir': {
        'models': ['StemGNN'],# 'TEFN','HELIX', 'HELIX_NoRotaryPE', 'HELIX_NoFeatureEmbed', 'HELIX_NoHybrid', 'HELIX_NoFusion','TimeMixerPP''ModernTCN','TOTEM','TimeMixer','TimeMixerPP',  'ImputeFormer', 
        'dataset_path': 'italy_air_quality_rate01_step12_point',
        'max_trials_per_model': 25,
    }
}

# ==================== 参数验证规则 ====================

def validate_params(model_name: str, params: Dict, dataset_name: str = None) -> tuple:
    """
    验证参数组合的合法性
    返回: (is_valid, reason)
    """
    
    # 通用规则：patch_stride <= patch_size
    if 'patch_size' in params and 'patch_stride' in params:
        if params['patch_stride'] > params['patch_size']:
            return False, f"patch_stride ({params['patch_stride']}) > patch_size ({params['patch_size']})"
        if params['patch_stride'] <= 0:
            return False, f"patch_stride must be > 0, got {params['patch_stride']}"
        if params['patch_size'] <= 0:
            return False, f"patch_size must be > 0, got {params['patch_size']}"
    
    # 通用规则：temperature必须严格大于最小阈值
    if 'temperature' in params:
        min_temp = 0.01
        if params['temperature'] <= 0 or params['temperature'] < min_temp:
            return False, f"temperature must be >= {min_temp}, got {params['temperature']}"
    
    # 通用规则：learning rate必须为正且不能太小或太大
    if 'learning_rate' in params or 'lr' in params:
        lr = params.get('learning_rate', params.get('lr'))
        min_lr = 1e-7
        max_lr = 0.1
        if lr <= 0 or lr < min_lr:
            return False, f"learning_rate must be >= {min_lr}, got {lr}"
        if lr > max_lr:
            return False, f"learning_rate too large (> {max_lr}), got {lr}"
    
    # 通用规则：dropout必须在[0, 1)范围内
    for dropout_key in ['dropout', 'attn_dropout', 'ff_dropout', 'head_dropout']:
        if dropout_key in params:
            dropout_val = params[dropout_key]
            if dropout_val < 0 or dropout_val >= 1:
                return False, f"{dropout_key} must be in [0, 1), got {dropout_val}"
    
    # 通用规则：d_model 必须为正
    if 'd_model' in params:
        if params['d_model'] <= 0:
            return False, f"d_model must be > 0, got {params['d_model']}"
    
    # 通用规则：n_layers 必须为正
    if 'n_layers' in params:
        if params['n_layers'] <= 0:
            return False, f"n_layers must be > 0, got {params['n_layers']}"
    
    # 通用规则：batch_size 必须为正
    if 'batch_size' in params:
        if params['batch_size'] <= 0:
            return False, f"batch_size must be > 0, got {params['batch_size']}"
    
    # 通用规则：d_kv 必须为正（用于计算 temperature = d_kv**0.5）
    if 'd_kv' in params:
        d_kv = params['d_kv']
        min_d_kv = 0.0001
        if d_kv <= 0:
            return False, f"d_kv must be > 0, got {d_kv}"
        if d_kv < min_d_kv:
            return False, f"d_kv must be >= {min_d_kv} (for temperature >= 0.01), got {d_kv}"
        temperature = d_kv ** 0.5
        if temperature < 0.01:
            return False, f"d_kv**0.5={temperature:.6f} < 0.01, need d_kv >= {min_d_kv}"
    
    # HELIX 特定规则
    if model_name == 'HELIX':
        if 'pe_dim' in params and params['pe_dim'] % 2 != 0:
            return False, f"HELIX: pe_dim ({params['pe_dim']}) must be even"
        if 'd_k' in params and params['d_k'] % 2 != 0:
            return False, f"HELIX: d_k ({params['d_k']}) must be even"
    
    # ModernTCN 特定规则
    if model_name == 'ModernTCN':
        if params.get('patch_stride', 0) > params.get('patch_size', float('inf')):
            return False, "ModernTCN: patch_stride must <= patch_size"
        
        dims_len = len(params.get('dims', []))
        num_blocks_len = len(params.get('num_blocks', []))
        large_size_len = len(params.get('large_size', []))
        small_size_len = len(params.get('small_size', []))
        
        if not (dims_len == num_blocks_len == large_size_len == small_size_len):
            return False, f"ModernTCN: dims({dims_len}), num_blocks({num_blocks_len}), large_size({large_size_len}), small_size({small_size_len}) must have same length"
        
        large_size = params.get('large_size', [])
        small_size = params.get('small_size', [])
        for i, (large, small) in enumerate(zip(large_size, small_size)):
            if small >= large:
                return False, f"ModernTCN: small_size[{i}] ({small}) must < large_size[{i}] ({large})"
            if small <= 0 or large <= 0:
                return False, f"ModernTCN: kernel sizes must be positive, got small_size[{i}]={small}, large_size[{i}]={large}"
    
    # TimeLLM 特定规则
    if model_name == 'TimeLLM':
        if params.get('patch_stride', 0) > params.get('patch_size', float('inf')):
            return False, "TimeLLM: patch_stride must <= patch_size"
        if 'temperature' in params and params['temperature'] < 0.01:
            return False, f"TimeLLM: temperature must be >= 0.01, got {params['temperature']}"
    
    # MOMENT 特定规则
    if model_name == 'MOMENT':
        if 'patch_size' in params and 'n_steps' in params:
            n_steps = params['n_steps']
            patch_size = params['patch_size']
            patch_stride = params.get('patch_stride', patch_size)
            
            if patch_size <= 0:
                return False, f"MOMENT: patch_size must be positive, got {patch_size}"
            if patch_stride <= 0:
                return False, f"MOMENT: patch_stride must be positive, got {patch_stride}"
            if patch_stride > patch_size:
                return False, f"MOMENT: patch_stride ({patch_stride}) must <= patch_size ({patch_size})"
            if patch_size > n_steps:
                return False, f"MOMENT: patch_size ({patch_size}) must <= n_steps ({n_steps})"
            
            n_patches = (n_steps - patch_size) // patch_stride + 1
            
            if n_patches < 2:
                return False, f"MOMENT: n_patches ({n_patches}) < 2"
            if n_patches > n_steps:
                return False, f"MOMENT: n_patches ({n_patches}) > n_steps ({n_steps})"
            
            last_patch_start = (n_patches - 1) * patch_stride
            if last_patch_start + patch_size > n_steps:
                return False, f"MOMENT: last patch exceeds sequence: {last_patch_start}+{patch_size} > {n_steps}"
            
            if 'd_model' in params:
                d_model = params['d_model']
                if d_model <= 0:
                    return False, f"MOMENT: d_model must be positive, got {d_model}"
                if d_model < patch_size:
                    return False, f"MOMENT: d_model ({d_model}) should be >= patch_size ({patch_size})"
    
    # PatchTST 特定规则
    if model_name == 'PatchTST':
        if 'patch_size' in params and 'n_steps' in params:
            n_steps = params['n_steps']
            patch_size = params['patch_size']
            patch_stride = params.get('patch_stride', patch_size)
            
            if patch_size <= 0:
                return False, f"PatchTST: patch_size must be positive, got {patch_size}"
            if patch_stride <= 0:
                return False, f"PatchTST: patch_stride must be positive, got {patch_stride}"
            if patch_stride > patch_size:
                return False, f"PatchTST: patch_stride ({patch_stride}) must <= patch_size ({patch_size})"
            if patch_size > n_steps:
                return False, f"PatchTST: patch_size ({patch_size}) must <= n_steps ({n_steps})"
            
            n_patches = (n_steps - patch_size) // patch_stride + 1
            if n_patches < 2:
                return False, f"PatchTST: n_patches ({n_patches}) < 2"
    
    # TimeMixer/TimeMixerPP 规则
    if model_name in ['TimeMixer', 'TimeMixerPP']:
        n_steps = params.get('n_steps', 48)
        downsampling_window = params.get('downsampling_window', 2)
        
        if downsampling_window <= 0:
            return False, f"{model_name}: downsampling_window must be > 0, got {downsampling_window}"
        if n_steps % downsampling_window != 0:
            return False, f"{model_name}: n_steps ({n_steps}) must be divisible by downsampling_window ({downsampling_window})"
        
        if 'd_kv' in params:
            d_kv = params['d_kv']
            min_d_kv = 0.0001
            if d_kv <= 0:
                return False, f"{model_name}: d_kv must be > 0, got {d_kv}"
            if d_kv < min_d_kv:
                return False, f"{model_name}: d_kv must be >= {min_d_kv}, got {d_kv}"
            temperature = d_kv ** 0.5
            if temperature < 0.01:
                return False, f"{model_name}: d_kv**0.5={temperature:.6f} < 0.01, need d_kv >= {min_d_kv}"
        
        if 'temperature' in params:
            if params['temperature'] <= 0 or params['temperature'] < 0.01:
                return False, f"{model_name}: temperature must be >= 0.01, got {params['temperature']}"
        
        if 'd_model' in params and 'n_heads' in params:
            d_model = params['d_model']
            n_heads = params['n_heads']
            if n_heads <= 0:
                return False, f"{model_name}: n_heads must be > 0, got {n_heads}"
            if d_model % n_heads != 0:
                return False, f"{model_name}: d_model ({d_model}) must be divisible by n_heads ({n_heads})"
    
    # TOTEM 规则
    if model_name == 'TOTEM':
        compression_factor = params.get('compression_factor', 4)
        if compression_factor not in [4, 8, 12, 16]:
            return False, f"TOTEM: compression_factor must be one of [4, 8, 12, 16], got {compression_factor}"
        
        n_steps = params.get('n_steps', 48)
        if n_steps % compression_factor != 0:
            return False, f"TOTEM: n_steps ({n_steps}) must be divisible by compression_factor ({compression_factor})"
        
        compressed_len = n_steps // compression_factor
        if compressed_len < 2:
            return False, f"TOTEM: compressed sequence length ({compressed_len}) is too short"
    
    # ImputeFormer 规则
    if model_name == 'ImputeFormer':
        if 'd_model' in params and 'n_heads' in params:
            d_model = params['d_model']
            n_heads = params['n_heads']
            if n_heads <= 0:
                return False, f"ImputeFormer: n_heads must be > 0, got {n_heads}"
            if d_model % n_heads != 0:
                return False, f"ImputeFormer: d_model ({d_model}) must be divisible by n_heads ({n_heads})"
    
    # TEFN 规则
    if model_name == 'TEFN':
        if 'd_model' in params and 'n_heads' in params:
            d_model = params['d_model']
            n_heads = params['n_heads']
            if n_heads <= 0:
                return False, f"TEFN: n_heads must be > 0, got {n_heads}"
            if d_model % n_heads != 0:
                return False, f"TEFN: d_model ({d_model}) must be divisible by n_heads ({n_heads})"
    
    return True, ""


# ==================== 工具函数 ====================

def sample_from_space(param_config: Dict) -> Any:
    """从参数配置中采样一个值"""
    param_type = param_config['_type']
    param_value = param_config['_value']
    
    if param_type == 'choice':
        return random.choice(param_value)
    elif param_type == 'loguniform':
        log_low = math.log(param_value[0])
        log_high = math.log(param_value[1])
        sampled = math.exp(random.uniform(log_low, log_high))
        # 避免采样到极小值（如temperature）
        if sampled < 1e-6:
            sampled = max(sampled, param_value[0])
        return sampled
    elif param_type == 'uniform':
        sampled = random.uniform(param_value[0], param_value[1])
        # 确保不会采样到边界外的值
        sampled = max(param_value[0], min(param_value[1], sampled))
        return sampled
    else:
        raise ValueError(f"Unknown parameter type: {param_type}")


def generate_random_params(model_name: str, tuning_space_path: str, num_trials: int, max_attempts: int = 1000, dataset_name: str = None) -> List[Dict]:
    """
    生成随机参数组合，并验证合法性
    max_attempts: 每个trial最多尝试采样的次数
    """
    with open(tuning_space_path, 'r') as f:
        tuning_space = json.load(f)
    
    param_combinations = []
    failed_validations = []
    
    for trial_id in range(num_trials):
        valid_params = None
        
        for attempt in range(max_attempts):
            # 采样参数
            params = {}
            for param_name, param_config in tuning_space.items():
                params[param_name] = sample_from_space(param_config)
            
            # 验证参数
            is_valid, reason = validate_params(model_name, params, dataset_name)
            
            if is_valid:
                valid_params = params
                break
            else:
                if attempt == 0:  # 只记录第一次失败的原因
                    failed_validations.append((trial_id + 1, reason))
        
        if valid_params is None:
            print(f"  ⚠️  警告: Trial {trial_id + 1} 在 {max_attempts} 次尝试后仍无法生成合法参数")
            # 使用最后一次采样的参数（即使不合法）
            valid_params = params
        
        param_combinations.append(valid_params)
    
    # 打印验证统计
    if failed_validations:
        print(f"  参数验证: {len(failed_validations)}/{num_trials} 个trials需要重新采样")
        # 显示前3个失败原因作为示例
        for trial_id, reason in failed_validations[:3]:
            print(f"    Trial {trial_id}: {reason}")
    
    return param_combinations

def format_param_value(value: Any) -> str:
    """格式化参数值为命令行参数"""
    if isinstance(value, bool):
        return str(value)
    elif isinstance(value, list):
        return '[' + ','.join(map(str, value)) + ']'
    elif isinstance(value, float):
        return f"{value:.6f}"
    else:
        return str(value)

def create_tuning_sbatch_script(
    model_name: str,
    dataset_name: str,
    dataset_path: str,
    params: Dict,
    trial_id: int,
    output_dir: str
) -> str:
    """创建单个trial的sbatch脚本"""
    
    # 将参数转换为命令行参数
    param_args = []
    for key, value in params.items():
        param_args.append(f"--{key} {format_param_value(value)}")
    
    param_str = " ".join(param_args)
    
    script_content = f"""#!/bin/bash
#SBATCH -x paraai-n32-h-01-agent-[1,4,8,16,17,25,27,28,29,30,31]
#SBATCH --job-name=tune_{model_name}_{dataset_name}_t{trial_id}
#SBATCH -o {output_dir}/trial_{trial_id}.out
#SBATCH -e {output_dir}/trial_{trial_id}.err
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
cat > {output_dir}/trial_{trial_id}_params.json <<'EOF'
{json.dumps(params, indent=2)}
EOF

# 运行训练
python -u train_model_tuning.py \\
    --model {model_name} \\
    --dataset {dataset_name} \\
    --dataset_fold_path {DATA_BASE_PATH}/{dataset_path} \\
    --saving_path {output_dir}/trial_{trial_id} \\
    --device cuda:0 \\
    --n_rounds 1 \\
    {param_str}

# 标记完成
echo "Trial {trial_id} completed at $(date)" >> {output_dir}/trial_{trial_id}_status.txt
"""
    
    return script_content


# ==================== 主函数 ====================

def main():
    random.seed(42)
    
    total_submitted = 0
    total_trials = 0
    
    # 计算总任务数
    for dataset_name, config in TUNING_CONFIG.items():
        total_trials += len(config['models']) * config['max_trials_per_model']
    
    print(f"{'='*70}")
    print(f"开始超参数调优任务提交")
    print(f"{'='*70}")
    print(f"总共需要提交: {total_trials} 个trials")
    
    for dataset_name in TUNING_CONFIG.keys():
        config = TUNING_CONFIG[dataset_name]
        print(f"{dataset_name}: {len(config['models'])} 模型 × {config['max_trials_per_model']} trials")
    
    print(f"{'='*70}\n")
    
    # 遍历每个数据集
    for dataset_name, config in TUNING_CONFIG.items():
        models = config['models']
        dataset_path = config['dataset_path']
        max_trials = config['max_trials_per_model']
        
        print(f"\n{'='*70}")
        print(f"处理数据集: {dataset_name}")
        print(f"{'='*70}")
        
        # 遍历每个模型
        for model_name in models:
            print(f"\n处理模型: {model_name}")
            print(f"{'-'*70}")
            
            # 读取调优空间
            tuning_space_path = os.path.join(
                BASE_DIR, CONFIG_BASE_PATH, model_name,
                f"{model_name}_{dataset_name}_tuning_space.json"
            )
            
            if not os.path.exists(tuning_space_path):
                print(f"⚠️  配置文件不存在: {tuning_space_path}")
                continue
            
            # 生成参数组合（带验证）
            print(f"生成 {max_trials} 组参数组合...")
            param_combinations = generate_random_params(model_name, tuning_space_path, max_trials, dataset_name=dataset_name)
            
            # 创建输出目录
            output_dir = os.path.join(
                BASE_DIR, TUNING_OUTPUT_PATH, 
                f"{model_name}_{dataset_name}_tuning"
            )
            os.makedirs(output_dir, exist_ok=True)
            
            # 提交每个trial
            submitted_this_model = 0
            failed_this_model = 0
            
            for trial_id, params in enumerate(param_combinations, 1):
                # 再次验证（保险起见）
                is_valid, reason = validate_params(model_name, params, dataset_name)
                if not is_valid:
                    print(f"  [{trial_id}/{max_trials}] ⚠️  跳过不合法参数: {reason}")
                    failed_this_model += 1
                    continue
                
                # 创建sbatch脚本
                script_content = create_tuning_sbatch_script(
                    model_name, dataset_name, dataset_path,
                    params, trial_id, output_dir
                )
                
                # 写入脚本文件
                script_path = os.path.join(output_dir, f"submit_trial_{trial_id}.sh")
                with open(script_path, 'w') as f:
                    f.write(script_content)
                
                # 提交任务
                try:
                    result = subprocess.run(
                        ['sbatch', script_path],
                        cwd=BASE_DIR,
                        capture_output=True,
                        text=True
                    )
                    
                    if result.returncode == 0 and 'Submitted batch job' in result.stdout:
                        job_id = result.stdout.strip()
                        print(f"  [{trial_id}/{max_trials}] ✓ {job_id}")
                        total_submitted += 1
                        submitted_this_model += 1
                    else:
                        print(f"  [{trial_id}/{max_trials}] ✗ 提交失败")
                        if result.stderr:
                            print(f"    错误: {result.stderr.strip()}")
                        failed_this_model += 1
                    
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"  [{trial_id}/{max_trials}] ✗ 异常: {e}")
                    failed_this_model += 1
            
            print(f"\n{model_name} 在 {dataset_name} 上:")
            print(f"  ✓ 成功提交: {submitted_this_model}/{max_trials}")
            if failed_this_model > 0:
                print(f"  ✗ 提交失败/跳过: {failed_this_model}/{max_trials}")
    
    print(f"\n{'='*70}")
    print(f"任务提交完成!")
    print(f"总共成功提交: {total_submitted}/{total_trials} 个trials")
    print(f"{'='*70}")
    print(f"\n检查任务状态: squeue -u $USER")
    print(f"查看结果: ls {TUNING_OUTPUT_PATH}/")


if __name__ == "__main__":
    main()