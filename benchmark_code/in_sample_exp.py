# 修改后的批量提交脚本 in_sample_exp.py
# -*- coding: utf-8 -*-
import os
import time
import subprocess


SKIP_COMBINATIONS = {
    # ItalyAir序列太短
    ('ItalyAir', 'MOMENT'),
    # PeMS特征太多(862) - TimeLLM/MOMENT/TimeMixerPP处理不了
    ('PeMS', 'MOMENT'),
    ('PeMS', 'TimeLLM'),
    ('PeMS', 'TimeMixerPP'),
    # Electricity特征太多(370) - TimeLLM/MOMENT/TimeMixerPP处理不了
    ('Electricity', 'MOMENT'),
    ('Electricity', 'TimeLLM'),
    ('Electricity', 'TimeMixerPP'),
}

DATASET_NAME_MAP = {
    'beijing_air_quality': 'BeijingAir',
    'electricity_load_diagrams': 'Electricity',
    'ett': 'ETT_h1',
    'italy_air_quality': 'ItalyAir',
    # Pedestrian单变量数据集 - 跳过所有新模型
    # 'melbourne_pedestrian': 'Pedestrian',
    'pems_traffic': 'PeMS',
    'physionet_2012': 'PhysioNet2012',
    'physionet_2019': 'PhysioNet2019',
}

MODEL_CONFIG_VERSIONS = {
    # 'HELIX': 'with_LR_decay',  # HELIX使用without_LR_decay版本
    # 其他模型默认为空字符串，即不添加后缀
}

DATA_BASE_PATH = "data/generated_datasets"
OUTPUT_BASE_PATH = "reproduce_imputation"
BASE_DIR = "/home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code"

# 模型列表
MODELS = [
    'HELIX',
    'TEFN',
    'TimeMixerPP',
    'TimeLLM',
    'MOMENT',
    'TimeMixer',
    'ModernTCN',
    'ImputeFormer',
    'TOTEM',
    # 'iTransformer',
    # 'SAITS',
    # 'FreTS',
    # 'NonstationaryTransformer',
    'PatchTST',
    
    
]

dataset_folders = [
    'beijing_air_quality_rate00_step24_block_blocklen6',
    'beijing_air_quality_rate01_step24_point',
    'beijing_air_quality_rate05_step24_point',
    'beijing_air_quality_rate05_step24_subseq_seqlen18',
    'beijing_air_quality_rate09_step24_point',
    'electricity_load_diagrams_rate00_step96_block_blocklen8',
    'electricity_load_diagrams_rate01_step96_point',
    'electricity_load_diagrams_rate05_step96_point',
    'electricity_load_diagrams_rate05_step96_subseq_seqlen72',
    'electricity_load_diagrams_rate09_step96_point',
    'ett_rate01_step48_point',
    'ett_rate03_step48_block_blocklen6',
    'ett_rate05_step48_point',
    'ett_rate05_step48_subseq_seqlen36',
    'ett_rate09_step48_point',
    'italy_air_quality_rate00_step12_block_blocklen4',
    'italy_air_quality_rate01_step12_point',
    'italy_air_quality_rate05_step12_point',
    'italy_air_quality_rate05_step12_subseq_seqlen8',
    'italy_air_quality_rate09_step12_point',
    'pems_traffic_rate00_step24_block_blocklen6',
    'pems_traffic_rate01_step24_point',
    'pems_traffic_rate05_step24_point',
    'pems_traffic_rate05_step24_subseq_seqlen18',
    'pems_traffic_rate09_step24_point',
    'physionet_2012_rate01_point',
]

def parse_dataset_info(folder_name):
    for key in DATASET_NAME_MAP.keys():
        if folder_name.startswith(key):
            dataset_name = DATASET_NAME_MAP[key]
            remaining = folder_name[len(key):]
            break
    else:
        return None
    
    rate_start = remaining.find('rate') + 4
    rate_end = remaining.find('_', rate_start)
    rate = remaining[rate_start:rate_end]
    
    if 'block' in folder_name:
        pattern = 'block'
    elif 'subseq' in folder_name:
        pattern = 'subseq'
    elif 'point' in folder_name:
        pattern = 'point'
    else:
        pattern = 'unknown'
    
    return dataset_name, rate, pattern

def create_sbatch_script(model_name, folder_name, dataset_name, rate, pattern, config_version):
    log_dir = f"{pattern}{rate}_log"
    dataset_log_dir = f"{log_dir}/{dataset_name}_log"
    
    version_flag = f"--config_version {config_version}" if config_version else ""
    version_suffix = f"_{config_version}" if config_version else ""
    
    script_content = f"""#!/bin/bash
#SBATCH -x paraai-n32-h-01-agent-[1,4,8,16,17,25,27,28,29,30,31]
#SBATCH --job-name={model_name}_{dataset_name}_{pattern}{rate}{version_suffix}
#SBATCH -o {OUTPUT_BASE_PATH}/{dataset_log_dir}/{model_name}_{dataset_name}{version_suffix}.out
#SBATCH -e {OUTPUT_BASE_PATH}/{dataset_log_dir}/{model_name}_{dataset_name}{version_suffix}.log
module purge
module load miniforge3/24.1 
module load compilers/cuda/12.1   compilers/gcc/11.3.0   cudnn/8.8.1.3_cuda12.x
source activate py310pots
export PYTHONUNBUFFERED=1
export http_proxy=http://u-cEoRwn:EDvFuZTe@172.16.4.9:3128
export https_proxy=http://u-cEoRwn:EDvFuZTe@172.16.4.9:3128   
export LD_PRELOAD=$LD_PRELOAD:/home/bingxing2/home/scx7644/.conda/envs/py310pots/lib/python3.10/site-packages/sklearn/utils/../../scikit_learn.libs/libgomp-947d5fa1.so.1.0.0
python -u train_model.py --model {model_name} --dataset {dataset_name} --dataset_fold_path {DATA_BASE_PATH}/{folder_name} --saving_path {OUTPUT_BASE_PATH}/{dataset_log_dir} --device cuda:0 {version_flag}
"""
    return script_content, dataset_log_dir, version_suffix

submitted_count = 0
skipped_count =0
total_tasks = len(MODELS) * len(dataset_folders)

print(f"准备提交 {len(MODELS)} 个新模型 × {len(dataset_folders)} 个数据集 = {total_tasks} 个任务\n")

for model_name in MODELS:
    for folder_name in dataset_folders:
        result = parse_dataset_info(folder_name)
        if result is None:
            print(f"Warning: Could not parse {folder_name}, skipping...")
            continue
        
        dataset_name, rate, pattern = result
        
        if (dataset_name, model_name) in SKIP_COMBINATIONS:
            print(f"[SKIP] {model_name} on {dataset_name} (incompatible)")
            skipped_count += 1
            continue
        
        config_version = MODEL_CONFIG_VERSIONS.get(model_name, "")
        
        script_content, dataset_log_dir, version_suffix = create_sbatch_script(model_name, folder_name, dataset_name, rate, pattern, config_version)
        
        output_dir = f"{OUTPUT_BASE_PATH}/{dataset_log_dir}"
        os.makedirs(output_dir, exist_ok=True)
        
        script_filename = f"{output_dir}/{model_name}_{dataset_name}{version_suffix}.sh"
        with open(script_filename, 'w') as f:
            f.write(script_content)
        
        try:
            result = subprocess.run(
                ['sbatch', '--gpus=1', script_filename],
                cwd=BASE_DIR,
                capture_output=True,
                text=True
            )
            print(f"[{submitted_count+1}/{total_tasks}] {model_name} on {folder_name}: {result.stdout.strip()}")
            if result.stderr:
                print(f"  Error: {result.stderr.strip()}")
            submitted_count += 1
            
            time.sleep(2)
        except Exception as e:
            print(f"Failed to submit {model_name} on {folder_name}: {e}")

print(f"\n总共提交了 {submitted_count}/{total_tasks} 个任务")