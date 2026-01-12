"""
Validate missing rate for each generated dataset.
"""

import os
import re
import h5py
import numpy as np
import pandas as pd


def parse_claimed_rate(dirname):
    """从目录名解析声称的缺失率"""
    match = re.search(r'rate(\d{2})', dirname)
    if match:
        rate_code = match.group(1)
        if rate_code == '01':
            return 0.10
        elif rate_code == '05':
            return 0.50
        elif rate_code == '09':
            return 0.90
        elif rate_code in ['00', '03']:
            return 0.50  # block 模式，声称 50%
    return np.nan


def calc_missing_rate(arr):
    """计算数组中 NaN 的占比"""
    return np.isnan(arr).sum() / arr.size


def validate_dataset(data_dir):
    """验证单个数据集的缺失率"""
    result = {}
    
    for split in ['train', 'val', 'test']:
        h5_path = os.path.join(data_dir, f'{split}.h5')
        if not os.path.exists(h5_path):
            result[f'{split}_missing'] = np.nan
            result[f'{split}_ori_missing'] = np.nan
            continue
            
        with h5py.File(h5_path, 'r') as hf:
            X = hf['X'][:]
            result[f'{split}_missing'] = calc_missing_rate(X)
            
            if 'X_ori' in hf.keys():
                X_ori = hf['X_ori'][:]
                result[f'{split}_ori_missing'] = calc_missing_rate(X_ori)
            else:
                result[f'{split}_ori_missing'] = np.nan
    
    return result


def main(base_dir, output_csv):
    """遍历所有数据集并生成报告"""
    records = []
    
    # 获取所有子目录
    subdirs = sorted([d for d in os.listdir(base_dir) 
                      if os.path.isdir(os.path.join(base_dir, d))])
    
    for dirname in subdirs:
        data_dir = os.path.join(base_dir, dirname)
        print(f"Processing: {dirname}")
        
        record = {
            'directory': dirname,
            'claimed_rate': parse_claimed_rate(dirname),
        }
        record.update(validate_dataset(data_dir))
        records.append(record)
    
    # 生成 DataFrame 并保存
    df = pd.DataFrame(records)
    df = df[['directory', 'claimed_rate', 
             'train_missing', 'val_missing', 'test_missing',
             'train_ori_missing', 'val_ori_missing', 'test_ori_missing']]
    
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")
    print(df.to_string())


if __name__ == "__main__":
    BASE_DIR = "/home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/data/generated_datasets"
    OUTPUT_CSV = "/home/bingxing2/home/scx7644/HELIX/Awesome_Imputation/benchmark_code/data/missing_rate_report.csv"
    
    main(BASE_DIR, OUTPUT_CSV)