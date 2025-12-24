import os
import shutil
from pathlib import Path

# 定义源目录和目标目录
source_base = "/home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/reproduce_imputation"
target_base = "/home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/results_process/imputation_log"

# 定义实验配置目录的映射关系（如果有不同的命名）
# 如果源目录和目标目录名称完全一致，可以省略映射
exp_mapping = {
    "block00_log": "block05_log",  # 根据实际情况调整
    "block03_log": "block05_log",  # 根据实际情况调整，或者可能需要其他映射
    "point01_log": "point01_log",
    "point05_log": "point05_log",
    "point09_log": "point09_log",
    "subseq05_log": "subseq05_log"
}

def copy_helix_logs():
    """拷贝HELIX模型的log文件到统一对比目录"""
    
    copied_count = 0
    skipped_count = 0
    
    # 遍历源目录中的实验配置目录
    for source_exp_dir in Path(source_base).iterdir():
        if not source_exp_dir.is_dir():
            continue
            
        exp_name = source_exp_dir.name
        
        # 检查是否在映射中
        if exp_name not in exp_mapping:
            print(f"跳过未映射的实验目录: {exp_name}")
            continue
            
        target_exp_name = exp_mapping[exp_name]
        target_exp_dir = Path(target_base) / target_exp_name
        
        # 检查目标实验目录是否存在
        if not target_exp_dir.exists():
            print(f"目标实验目录不存在: {target_exp_dir}")
            continue
        
        print(f"\n处理实验配置: {exp_name} -> {target_exp_name}")
        
        # 遍历数据集目录
        for dataset_dir in source_exp_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
                
            dataset_name = dataset_dir.name
            target_dataset_dir = target_exp_dir / dataset_name
            
            # 创建目标数据集目录（如果不存在）
            target_dataset_dir.mkdir(parents=True, exist_ok=True)
            
            # 查找HELIX的.log文件
            helix_log_files = list(dataset_dir.glob("HELIX_*.log"))
            
            for log_file in helix_log_files:
                target_log_file = target_dataset_dir / log_file.name
                
                try:
                    shutil.copy2(log_file, target_log_file)
                    print(f"  ✓ 拷贝: {log_file.name} -> {target_dataset_dir.name}/")
                    copied_count += 1
                except Exception as e:
                    print(f"  ✗ 拷贝失败 {log_file.name}: {e}")
                    skipped_count += 1
    
    print(f"\n{'='*60}")
    print(f"拷贝完成！成功: {copied_count} 个文件，跳过/失败: {skipped_count} 个文件")
    print(f"{'='*60}")

if __name__ == "__main__":
    copy_helix_logs()