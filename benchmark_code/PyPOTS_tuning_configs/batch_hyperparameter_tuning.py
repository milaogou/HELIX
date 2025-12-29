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
        'models': [ ],#'HELIX', 'TEFN', 'TimeMixer', 'ImputeFormer','ModernTCN',   'TOTEM', 'TimeMixerPP', 'TimeLLM', 'MOMENT'
        'dataset_path': 'ett_rate01_step48_point',
        'max_trials_per_model': 25,
    },
    'PeMS': {
        'models': [ ],#'HELIX', 'TEFN', 'TimeMixer','ImputeFormer','ModernTCN',  'TOTEM'
        'dataset_path': 'pems_traffic_rate01_step24_point',
        'max_trials_per_model': 25,
    },
    'BeijingAir': {
        'models': [],#'HELIX', 'TEFN', 'TimeMixer', 'ModernTCN', 'ImputeFormer', 'TOTEM', 'TimeMixerPP', 'MOMENT'
        'dataset_path': 'beijing_air_quality_rate01_step24_point',
        'max_trials_per_model': 25,
    },
    'PhysioNet2012': {
        'models': [],#'HELIX', 'TEFN', 'TimeMixer', 'ModernTCN', 'ImputeFormer', 'TOTEM', 'TimeMixerPP', 'MOMENT'
        'dataset_path': 'physionet_2012_rate01_point',
        'max_trials_per_model': 25,
    },
    'ItalyAir': {
        'models': ['HELIX', 'TEFN', 'TimeMixer', 'ModernTCN', 'ImputeFormer', 'TOTEM', 'TimeMixerPP'],#
        'dataset_path': 'italy_air_quality_rate01_step12_point',
        'max_trials_per_model': 25,
    }
}

# ==================== 参数验证规则 ====================

def validate_params(model_name: str, params: Dict) -> tuple:
    """
    验证参数组合的合法性
    返回: (is_valid, reason)
    """
    
    # 通用规则：patch_stride <= patch_size
    if 'patch_size' in params and 'patch_stride' in params:
        if params['patch_stride'] > params['patch_size']:
            return False, f"patch_stride ({params['patch_stride']}) > patch_size ({params['patch_size']})"
    
    # ModernTCN 特定规则
    if model_name == 'ModernTCN':
        # patch_stride 必须 <= patch_size
        if params.get('patch_stride', 0) > params.get('patch_size', float('inf')):
            return False, "ModernTCN: patch_stride must <= patch_size"
        
        # 检查 dims 和 num_blocks 长度一致
        if len(params.get('dims', [])) != len(params.get('num_blocks', [])):
            return False, "ModernTCN: dims and num_blocks must have same length"
    
    # TimeLLM 特定规则
    if model_name == 'TimeLLM':
        # patch_stride 必须 <= patch_size
        if params.get('patch_stride', 0) > params.get('patch_size', float('inf')):
            return False, "TimeLLM: patch_stride must <= patch_size"
    
    # MOMENT 特定规则
    if model_name == 'MOMENT':
        # patch_stride 必须 <= patch_size
        if params.get('patch_stride', 0) > params.get('patch_size', float('inf')):
            return False, "MOMENT: patch_stride must <= patch_size"
    
    # PatchTST 特定规则
    if model_name == 'PatchTST':
        # patch_stride 必须 <= patch_size
        if params.get('patch_stride', 0) > params.get('patch_size', float('inf')):
            return False, "PatchTST: patch_stride must <= patch_size"
    
    # TimeMixer/TimeMixerPP 规则
    if model_name in ['TimeMixer', 'TimeMixerPP']:
        # downsampling_window 必须能整除 n_steps
        n_steps = params.get('n_steps', 48)
        downsampling_window = params.get('downsampling_window', 2)
        if n_steps % downsampling_window != 0:
            return False, f"{model_name}: n_steps ({n_steps}) must be divisible by downsampling_window ({downsampling_window})"
    
    # TOTEM 规则
    if model_name == 'TOTEM':
        # compression_factor 必须能整除 n_steps
        n_steps = params.get('n_steps', 48)
        compression_factor = params.get('compression_factor', 4)
        if n_steps % compression_factor != 0:
            return False, f"TOTEM: n_steps ({n_steps}) must be divisible by compression_factor ({compression_factor})"
    
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
        return math.exp(random.uniform(log_low, log_high))
    elif param_type == 'uniform':
        return random.uniform(param_value[0], param_value[1])
    else:
        raise ValueError(f"Unknown parameter type: {param_type}")


def generate_random_params(model_name: str, tuning_space_path: str, num_trials: int, max_attempts: int = 100) -> List[Dict]:
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
            is_valid, reason = validate_params(model_name, params)
            
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
            param_combinations = generate_random_params(model_name, tuning_space_path, max_trials)
            
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
                is_valid, reason = validate_params(model_name, params)
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