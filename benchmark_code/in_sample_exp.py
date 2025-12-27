# -*- coding: utf-8 -*-
import os
import time
import subprocess

# 数据集文件夹到简化名称的映射
DATASET_NAME_MAP = {
    'beijing_air_quality': 'BeijingAir',
    'electricity_load_diagrams': 'Electricity',
    'ett': 'ETT_h1',
    'italy_air_quality': 'ItalyAir',
    'melbourne_pedestrian': 'Pedestrian',
    'pems_traffic': 'PeMS',
    'physionet_2012': 'PhysioNet2012',
    'physionet_2019': 'PhysioNet2019',
}

# 数据集基础路径
DATA_BASE_PATH = "data/generated_datasets"
OUTPUT_BASE_PATH = "reproduce_imputation"
BASE_DIR = "/home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code"

# 获取所有数据集文件夹
dataset_folders = [
    'beijing_air_quality_rate00_step24_block_blocklen6',
    # 'beijing_air_quality_rate01_step24_point',
    'beijing_air_quality_rate05_step24_point',
    'beijing_air_quality_rate05_step24_subseq_seqlen18',
    'beijing_air_quality_rate09_step24_point',
    # 'electricity_load_diagrams_rate00_step96_block_blocklen8',
    # 'electricity_load_diagrams_rate01_step96_point',
    # 'electricity_load_diagrams_rate05_step96_point',
    # 'electricity_load_diagrams_rate05_step96_subseq_seqlen72',
    # 'electricity_load_diagrams_rate09_step96_point',
    # 'ett_rate01_step48_point',
    # 'ett_rate03_step48_block_blocklen6',
    # 'ett_rate05_step48_point',
    # 'ett_rate05_step48_subseq_seqlen36',
    # 'ett_rate09_step48_point',
    # 'italy_air_quality_rate00_step12_block_blocklen4',
    # 'italy_air_quality_rate01_step12_point',
    # 'italy_air_quality_rate05_step12_point',
    # 'italy_air_quality_rate05_step12_subseq_seqlen8',
    # 'italy_air_quality_rate09_step12_point',
    # 'melbourne_pedestrian_rate01_step24_point',
    # 'melbourne_pedestrian_rate05_step24_point',
    # 'melbourne_pedestrian_rate09_step24_point',
    # 'pems_traffic_rate00_step24_block_blocklen6',
    # 'pems_traffic_rate01_step24_point',
    # 'pems_traffic_rate05_step24_point',
    # 'pems_traffic_rate05_step24_subseq_seqlen18',
    # 'pems_traffic_rate09_step24_point',
    # 'physionet_2012_rate01_point',
]

def parse_dataset_info(folder_name):
    """解析数据集文件夹名，提取数据集名称、rate和pattern"""
    # 提取数据集前缀
    for key in DATASET_NAME_MAP.keys():
        if folder_name.startswith(key):
            dataset_name = DATASET_NAME_MAP[key]
            remaining = folder_name[len(key):]
            break
    else:
        return None
    
    # 提取rate
    rate_start = remaining.find('rate') + 4
    rate_end = remaining.find('_', rate_start)
    rate = remaining[rate_start:rate_end]
    
    # 提取pattern
    if 'block' in folder_name:
        pattern = 'block'
    elif 'subseq' in folder_name:
        pattern = 'subseq'
    elif 'point' in folder_name:
        pattern = 'point'
    else:
        pattern = 'unknown'
    
    return dataset_name, rate, pattern

def create_sbatch_script(folder_name, dataset_name, rate, pattern):
    """生成sbatch脚本内容"""
    log_dir = f"{pattern}{rate}_log"
    dataset_log_dir = f"{log_dir}/{dataset_name}_log"
    
    script_content = f"""#!/bin/bash
#SBATCH -x paraai-n32-h-01-agent-[1,4,8,16,17,25,27,28,29,30,31]
#SBATCH --job-name=HELIX_{dataset_name}_{pattern}{rate}
#SBATCH -o {OUTPUT_BASE_PATH}/{dataset_log_dir}/HELIX_{dataset_name}.out
#SBATCH -e {OUTPUT_BASE_PATH}/{dataset_log_dir}/HELIX_{dataset_name}.log
module purge
module load miniforge3/24.1 
module load compilers/cuda/12.1   compilers/gcc/11.3.0   cudnn/8.8.1.3_cuda12.x
source activate py310pots
export PYTHONUNBUFFERED=1   
python -u train_model.py --model HELIX --dataset {dataset_name} --dataset_fold_path {DATA_BASE_PATH}/{folder_name} --saving_path {OUTPUT_BASE_PATH}/{dataset_log_dir} --device cuda:0
"""
    return script_content, dataset_log_dir

# 创建所有任务的脚本并提交
submitted_count = 0
for folder_name in dataset_folders:
    result = parse_dataset_info(folder_name)
    if result is None:
        print(f"Warning: Could not parse {folder_name}, skipping...")
        continue
    
    dataset_name, rate, pattern = result
    
    # 生成脚本
    script_content, dataset_log_dir = create_sbatch_script(folder_name, dataset_name, rate, pattern)
    
    # 创建输出目录
    output_dir = f"{OUTPUT_BASE_PATH}/{dataset_log_dir}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created directory: {output_dir}")
    
    # 保存脚本文件
    script_filename = f"{output_dir}/HELIX_{dataset_name}.sh"
    with open(script_filename, 'w') as f:
        f.write(script_content)
    print(f"Created script: {script_filename}")
    
    # 提交任务
    try:
        result = subprocess.run(
            ['sbatch', '--gpus=1', script_filename],
            cwd=BASE_DIR,
            capture_output=True,
            text=True
        )
        print(f"Submitted: {folder_name}")
        print(f"  Output: {result.stdout.strip()}")
        if result.stderr:
            print(f"  Error: {result.stderr.strip()}")
        submitted_count += 1
        
        # 等待2秒再提交下一个
        if submitted_count < len(dataset_folders):
            time.sleep(2)
    except Exception as e:
        print(f"Failed to submit {folder_name}: {e}")

print(f"\n总共提交了 {submitted_count} 个任务")