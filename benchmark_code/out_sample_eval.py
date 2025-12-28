"""
Out-of-sample evaluation script
Evaluate models trained on one missing pattern on test sets with different missing patterns
"""

import argparse
import os
import glob
import numpy as np
import torch
from pypots.data.saving import pickle_dump
from pypots.imputation import *
from pypots.utils.logging import logger
from pypots.utils.metrics import calc_mae, calc_mse, calc_mre

from global_config import TORCH_N_THREADS
from utils import get_datasets_path
from hpo_results import HPO_RESULTS

SUPPORT_MODELS = {
    "ImputeFormer": ImputeFormer,
    "TEFN": TEFN,
    "SAITS": SAITS,
    "iTransformer": iTransformer,
    "HELIX": HELIX,
}

def find_model_file(round_dir, model_name):
    """Find the actual .pypots model file in the timestamped subdirectory"""
    if not os.path.exists(round_dir):
        return None
    
    # Find all timestamped directories (format: YYYYMMDD_THHMMSS)
    timestamp_dirs = []
    for d in os.listdir(round_dir):
        full_path = os.path.join(round_dir, d)
        if os.path.isdir(full_path) and d.startswith('202'):
            timestamp_dirs.append(d)
    
    if not timestamp_dirs:
        logger.error(f"No timestamped directory found in {round_dir}")
        return None
    
    # Sort by timestamp (newest first) to handle multiple training runs
    timestamp_dirs.sort(reverse=True)
    
    if len(timestamp_dirs) > 1:
        logger.warning(f"Found {len(timestamp_dirs)} timestamped directories in {round_dir}, using the latest: {timestamp_dirs[0]}")
    
    timestamp_dir = os.path.join(round_dir, timestamp_dirs[0])
    
    # Look for .pypots file
    model_file = os.path.join(timestamp_dir, f"{model_name}.pypots")
    
    if os.path.exists(model_file):
        return model_file
    else:
        logger.error(f"Model file not found: {model_file}")
        return None

def load_model_from_file(model_class, model_file, dataset_name, device):
    """Load a PyPOTS model from .pypots file
    
    This function recreates the model with the same hyperparameters and loads the trained weights.
    """
    # Load the saved checkpoint
    checkpoint = torch.load(model_file, map_location='cpu')
    
    logger.info(f"Checkpoint keys: {checkpoint.keys()}")
    
    # Get hyperparameters from HPO_RESULTS
    hyperparameters = HPO_RESULTS[dataset_name][model_class.__name__].copy()
    
    # Remove learning rate as we don't need optimizer for inference
    if 'lr' in hyperparameters:
        hyperparameters.pop('lr')
    
    # Set device and other inference-specific settings
    hyperparameters['device'] = device
    hyperparameters['saving_path'] = None  # Don't save during evaluation
    hyperparameters['model_saving_strategy'] = None
    
    # Remove optimizer-related params if exist
    hyperparameters.pop('G_optimizer', None)
    hyperparameters.pop('D_optimizer', None)
    hyperparameters.pop('optimizer', None)
    
    logger.info(f"Recreating model with hyperparameters: {list(hyperparameters.keys())}")
    
    # Create model instance
    model = model_class(**hyperparameters)
    
    # Load model state dict
    model_state_dict = checkpoint.get('model_state_dict', None)
    if model_state_dict is None:
        # Try loading old format (pypots < 0.13)
        logger.warning("Old format checkpoint detected, trying to load directly...")
        if hasattr(checkpoint, 'state_dict'):
            model_state_dict = checkpoint.state_dict()
        else:
            raise ValueError("Cannot find model_state_dict in checkpoint")
    
    # Load weights
    if hasattr(model, 'model'):
        model.model.load_state_dict(model_state_dict)
        model.model.eval()
    else:
        raise ValueError("Model does not have 'model' attribute")
    
    logger.info("Model loaded and set to eval mode")
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=list(SUPPORT_MODELS.keys()))
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--train_pattern", type=str, default="point05", help="Training pattern (e.g., point05)")
    parser.add_argument("--test_pattern", type=str, required=True, help="Test pattern folder name")
    parser.add_argument("--round_id", type=int, required=True, help="Training round ID (0-4)")
    parser.add_argument("--model_base_path", type=str, default="reproduce_imputation")
    parser.add_argument("--data_base_path", type=str, default="data/generated_datasets")
    parser.add_argument("--output_base_path", type=str, default="out_sample_eval")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    torch.set_num_threads(TORCH_N_THREADS)
    
    # Construct model directory path
    train_log_dir = f"{args.train_pattern}_log"
    dataset_log_dir = f"{train_log_dir}/{args.dataset}_log"
    model_round_dir = os.path.join(
        args.model_base_path, 
        dataset_log_dir,
        f"{args.model}_{args.dataset}",
        f"round_{args.round_id}"
    )
    
    # Find actual model file (.pypots)
    model_file = find_model_file(model_round_dir, args.model)
    if model_file is None:
        logger.error(f"Model file not found in {model_round_dir}")
        exit(1)
    
    logger.info(f"Found model file: {model_file}")
    
    # Load test data
    test_data_path = os.path.join(args.data_base_path, args.test_pattern)
    if not os.path.exists(test_data_path):
        logger.error(f"Test data path not found: {test_data_path}")
        exit(1)
    
    _, _, test_X, test_X_ori, test_indicating_mask = get_datasets_path(test_data_path)
    
    # Load trained model
    logger.info(f"Loading model from {model_file}")
    model_class = SUPPORT_MODELS[args.model]
    
    try:
        model = load_model_from_file(model_class, model_file, args.dataset, args.device)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # Predict on test set
    test_set = {"X": test_X}
    logger.info(f"Evaluating {args.model} (trained on {args.train_pattern}) on test set {args.test_pattern}")
    
    try:
        if args.model in ["CSDI", "GPVAE"]:
            results = model.predict(test_set, n_sampling_times=10)
            test_set_imputation = results["imputation"].mean(axis=1)
        else:
            results = model.predict(test_set)
            test_set_imputation = results["imputation"]
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # Calculate metrics
    mae = calc_mae(test_set_imputation, test_X_ori, test_indicating_mask)
    mse = calc_mse(test_set_imputation, test_X_ori, test_indicating_mask)
    mre = calc_mre(test_set_imputation, test_X_ori, test_indicating_mask)
    
    # Save results
    output_dir = os.path.join(
        args.output_base_path,
        f"{args.model}_{args.dataset}_train_{args.train_pattern}_test_{args.test_pattern}",
        f"round_{args.round_id}"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    results_dict = {
        "mae": mae,
        "mse": mse,
        "mre": mre,
        "model": args.model,
        "dataset": args.dataset,
        "train_pattern": args.train_pattern,
        "test_pattern": args.test_pattern,
        "round_id": args.round_id,
    }
    
    pickle_dump(results_dict, os.path.join(output_dir, "results.pkl"))
    
    logger.info(
        f"Results saved to {output_dir}\n"
        f"{args.model} (trained on {args.train_pattern}) on {args.test_pattern} test set:\n"
        f"MAE={mae:.4f}, MSE={mse:.4f}, MRE={mre:.4f}"
    )
    
    # Also save as text for easy reading
    with open(os.path.join(output_dir, "results.txt"), "w") as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Train Pattern: {args.train_pattern}\n")
        f.write(f"Test Pattern: {args.test_pattern}\n")
        f.write(f"Round ID: {args.round_id}\n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"MSE: {mse:.4f}\n")
        f.write(f"MRE: {mre:.4f}\n")