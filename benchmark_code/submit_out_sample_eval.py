"""
Batch submission script for out-of-sample evaluation
"""

import os
import time
import subprocess

BASE_DIR = "/home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code"
MODEL_BASE_PATH = "reproduce_imputation"
DATA_BASE_PATH = "data/generated_datasets"
OUTPUT_BASE_PATH = "out_sample_eval"

# Models to evaluate
MODELS = ['TEFN', 'SAITS', 'iTransformer']#,'ImputeFormer', 'HELIX'

# Dataset patterns mapping
DATASET_PATTERNS = {
    # 'BeijingAir': [
    #     'beijing_air_quality_rate00_step24_block_blocklen6',
    #     'beijing_air_quality_rate01_step24_point',
    #     'beijing_air_quality_rate05_step24_point',
    #     'beijing_air_quality_rate05_step24_subseq_seqlen18',
    #     'beijing_air_quality_rate09_step24_point',
    # ],
    # 'Electricity': [
    #     'electricity_load_diagrams_rate00_step96_block_blocklen8',
    #     'electricity_load_diagrams_rate01_step96_point',
    #     'electricity_load_diagrams_rate05_step96_point',
    #     'electricity_load_diagrams_rate05_step96_subseq_seqlen72',
    #     'electricity_load_diagrams_rate09_step96_point',
    # ],
    # 'ETT_h1': [
    #     'ett_rate01_step48_point',
    #     'ett_rate03_step48_block_blocklen6',
    #     'ett_rate05_step48_point',
    #     'ett_rate05_step48_subseq_seqlen36',
    #     'ett_rate09_step48_point',
    # ],
    # 'ItalyAir': [
    #     'italy_air_quality_rate00_step12_block_blocklen4',
    #     'italy_air_quality_rate01_step12_point',
    #     'italy_air_quality_rate05_step12_point',
    #     'italy_air_quality_rate05_step12_subseq_seqlen8',
    #     'italy_air_quality_rate09_step12_point',
    # ],
    'PeMS': [
        'pems_traffic_rate00_step24_block_blocklen6',
        'pems_traffic_rate01_step24_point',
        'pems_traffic_rate05_step24_point',
        'pems_traffic_rate05_step24_subseq_seqlen18',
        'pems_traffic_rate09_step24_point',
    ]
}

# Training pattern (all models trained on point05)
TRAIN_PATTERNS = {
    'BeijingAir': 'point05',
    'Electricity': 'point05',
    'ETT_h1': 'point05',
    'ItalyAir': 'point05',
    'PeMS': 'point05',
}

# Number of rounds (0-4)
N_ROUNDS = 5

def get_pattern_short_name(pattern_full):
    """Extract short pattern name for logging"""
    # Handle short format like "point05", "block00", "subseq05"
    if not '_' in pattern_full or 'rate' not in pattern_full:
        # Already in short format or simple pattern name
        if 'point' in pattern_full:
            # Extract number after 'point'
            num = pattern_full.replace('point', '')
            return f'point{num}'
        elif 'block' in pattern_full:
            num = pattern_full.replace('block', '')
            return f'block{num}'
        elif 'subseq' in pattern_full:
            num = pattern_full.replace('subseq', '')
            return f'subseq{num}'
        else:
            return pattern_full
    
    # Handle full format like "beijing_air_quality_rate05_step24_point"
    if 'block' in pattern_full:
        rate = pattern_full.split('rate')[1].split('_')[0]
        return f'block{rate}'
    elif 'subseq' in pattern_full:
        rate = pattern_full.split('rate')[1].split('_')[0]
        return f'subseq{rate}'
    elif 'point' in pattern_full:
        rate = pattern_full.split('rate')[1].split('_')[0]
        return f'point{rate}'
    return 'unknown'

def create_sbatch_script(model, dataset, train_pattern, test_pattern, round_id):
    """Create sbatch script for evaluation job"""
    test_short = get_pattern_short_name(test_pattern)
    train_short = get_pattern_short_name(train_pattern)
    
    log_dir = os.path.join(OUTPUT_BASE_PATH, "logs", dataset)
    os.makedirs(log_dir, exist_ok=True)
    
    job_name = f"{model}_{dataset}_tr{train_short}_te{test_short}_r{round_id}"
    
    script_content = f"""#!/bin/bash
#SBATCH -x paraai-n32-h-01-agent-[1,4,8,16,17,25,27,28,29,30,31]
#SBATCH --job-name={job_name}
#SBATCH -o {log_dir}/{job_name}.out
#SBATCH -e {log_dir}/{job_name}.log
#SBATCH --gpus=1

module purge
module load miniforge3/24.1 
module load compilers/cuda/12.1 compilers/gcc/11.3.0 cudnn/8.8.1.3_cuda12.x
source activate py310pots

export PYTHONUNBUFFERED=1
export http_proxy=http://u-cEoRwn:EDvFuZTe@172.16.4.9:3128
export https_proxy=http://u-cEoRwn:EDvFuZTe@172.16.4.9:3128
export LD_PRELOAD=$LD_PRELOAD:/home/bingxing2/home/scx7644/.conda/envs/py310pots/lib/python3.10/site-packages/sklearn/utils/../../scikit_learn.libs/libgomp-947d5fa1.so.1.0.0
python -u out_sample_eval.py \\
    --model {model} \\
    --dataset {dataset} \\
    --train_pattern {train_pattern} \\
    --test_pattern {test_pattern} \\
    --round_id {round_id} \\
    --model_base_path {MODEL_BASE_PATH} \\
    --data_base_path {DATA_BASE_PATH} \\
    --output_base_path {OUTPUT_BASE_PATH} \\
    --device cuda:0
"""
    return script_content, job_name

def main():
    submitted_count = 0
    skipped_count = 0
    
    # Calculate total tasks
    total_tasks = 0
    for dataset, patterns in DATASET_PATTERNS.items():
        total_tasks += len(MODELS) * len(patterns) * N_ROUNDS
    
    print(f"准备提交Out-of-Sample评估任务")
    print(f"=" * 80)
    print(f"模型数量: {len(MODELS)}")
    print(f"数据集数量: {len(DATASET_PATTERNS)}")
    print(f"每个数据集的测试模式: 1-5个")
    print(f"每个配置的轮次: {N_ROUNDS}")
    print(f"预计总任务数: ~{total_tasks}")
    print(f"=" * 80)
    print()
    
    for dataset, patterns in DATASET_PATTERNS.items():
        print(f"\n{'='*80}")
        print(f"处理数据集: {dataset}")
        print(f"训练模式: {TRAIN_PATTERNS[dataset]}")
        print(f"测试模式数量: {len(patterns)}")
        print(f"{'='*80}")
        
        train_pattern = TRAIN_PATTERNS[dataset]
        
        for model in MODELS:
            print(f"\n  模型: {model}")
            
            # Check if trained model exists
            train_log_dir = f"{train_pattern}_log"
            model_check_path = os.path.join(
                MODEL_BASE_PATH,
                train_log_dir,
                f"{dataset}_log",
                f"{model}_{dataset}",
                "round_0"
            )
            
            if not os.path.exists(model_check_path):
                print(f"    [SKIP] 训练模型不存在: {model_check_path}")
                skipped_count += len(patterns) * N_ROUNDS
                continue
            
            for test_pattern in patterns:
                test_short = get_pattern_short_name(test_pattern)
                
                for round_id in range(N_ROUNDS):
                    script_content, job_name = create_sbatch_script(
                        model, dataset, train_pattern, test_pattern, round_id
                    )
                    
                    # Save script
                    script_dir = os.path.join(OUTPUT_BASE_PATH, "scripts", dataset)
                    os.makedirs(script_dir, exist_ok=True)
                    script_file = os.path.join(script_dir, f"{job_name}.sh")
                    
                    with open(script_file, 'w') as f:
                        f.write(script_content)
                    
                    # Submit job
                    try:
                        result = subprocess.run(
                            ['sbatch', script_file],
                            cwd=BASE_DIR,
                            capture_output=True,
                            text=True
                        )
                        
                        submitted_count += 1
                        if submitted_count % 10 == 0 or submitted_count <= 5:
                            print(f"    [{submitted_count}] {test_short} Round{round_id}: {result.stdout.strip()}")
                        
                        if result.stderr:
                            print(f"      Error: {result.stderr.strip()}")
                        
                        time.sleep(0.5)  # Avoid overwhelming the scheduler
                        
                    except Exception as e:
                        print(f"    [ERROR] Failed to submit {job_name}: {e}")
            
            print(f"    ✓ {model} 完成提交")
    
    print(f"\n{'='*80}")
    print(f"提交完成!")
    print(f"成功提交: {submitted_count} 个任务")
    print(f"跳过: {skipped_count} 个任务")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()