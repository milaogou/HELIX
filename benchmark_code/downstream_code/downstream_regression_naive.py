"""
Downstream regression task for naive imputation methods.
Adapted for naive_imputation.h5 format (mean, median, locf, linear_interpolation).
"""

import argparse
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from global_config import RANDOM_SEEDS


class RegressionRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)
    
    def forward(self, x):
        output, _ = self.rnn(x)
        return self.fc(output).squeeze(-1)


class RegressionTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.fc(x).squeeze(-1)


def train_neural_regressor(model, train_loader, val_loader, epochs=50, lr=1e-3, device='cpu'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                val_loss += criterion(outputs, y_batch).item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
    
    model.load_state_dict(best_model_state)
    return model


def evaluate_neural_regressor(model, test_loader, device='cpu'):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(y_batch.numpy())
    
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    
    mae = mean_absolute_error(labels.flatten(), preds.flatten())
    mse = mean_squared_error(labels.flatten(), preds.flatten())
    mre = np.mean(np.abs(labels - preds) / (np.abs(labels) + 1e-8))
    
    return mae, mse, mre


def run_regression(X_train, y_train, X_val, y_val, X_test, y_test, seed):
    results = {}
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Flatten for XGBoost
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    y_train_flat = y_train.flatten()
    y_test_flat = y_test.flatten()
    
    # Standardize
    scaler_X = StandardScaler()
    X_train_flat = scaler_X.fit_transform(X_train_flat)
    X_val_flat = scaler_X.transform(X_val_flat)
    X_test_flat = scaler_X.transform(X_test_flat)
    
    # Repeat X for each timestep
    n_steps = y_train.shape[1]
    X_train_rep = np.repeat(X_train_flat, n_steps, axis=0)
    X_test_rep = np.repeat(X_test_flat, n_steps, axis=0)
    
    # XGBoost
    xgb = XGBRegressor(n_estimators=100, max_depth=5, random_state=seed)
    xgb.fit(X_train_rep, y_train_flat)
    preds_xgb = xgb.predict(X_test_rep)
    
    results['XGB_MAE'] = mean_absolute_error(y_test_flat, preds_xgb)
    results['XGB_MSE'] = mean_squared_error(y_test_flat, preds_xgb)
    results['XGB_MRE'] = np.mean(np.abs(y_test_flat - preds_xgb) / (np.abs(y_test_flat) + 1e-8))
    
    # Prepare data for neural models
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=64)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=64)
    
    input_dim = X_train.shape[-1]
    
    # RNN
    rnn = RegressionRNN(input_dim)
    rnn = train_neural_regressor(rnn, train_loader, val_loader)
    results['RNN_MAE'], results['RNN_MSE'], results['RNN_MRE'] = evaluate_neural_regressor(rnn, test_loader)
    
    # Transformer
    transformer = RegressionTransformer(input_dim)
    transformer = train_neural_regressor(transformer, train_loader, val_loader)
    results['Transformer_MAE'], results['Transformer_MSE'], results['Transformer_MRE'] = evaluate_neural_regressor(transformer, test_loader)
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_fold_path", type=str, required=True)
    parser.add_argument("--naive_method", type=str, required=True, 
                        choices=["mean", "median", "locf", "linear_interpolation"])
    parser.add_argument("--n_rounds", type=int, default=5)
    args = parser.parse_args()
    
    print(f"Running downstream regression for Naive_{args.naive_method}")
    print(f"Dataset: {args.dataset_fold_path}")
    
    # Load original data for targets
    with h5py.File(f"{args.dataset_fold_path}/train.h5", "r") as f:
        X_ori_train = f["X_ori"][:]
    with h5py.File(f"{args.dataset_fold_path}/val.h5", "r") as f:
        X_ori_val = f["X_ori"][:]
    with h5py.File(f"{args.dataset_fold_path}/test.h5", "r") as f:
        X_ori_test = f["X_ori"][:]
    
    # Target: last feature for all timesteps
    y_train = X_ori_train[:, :, -1]
    y_val = X_ori_val[:, :, -1]
    y_test = X_ori_test[:, :, -1]
    
    # Load naive imputation results (exclude last feature)
    with h5py.File(f"{args.dataset_fold_path}/naive_imputation.h5", "r") as f:
        X_train = f[f"train/{args.naive_method}"][:][:, :, :-1]
        X_val = f[f"val/{args.naive_method}"][:][:, :, :-1]
        X_test = f[f"test/{args.naive_method}"][:][:, :, :-1]
    
    all_results = []
    for round_idx in range(args.n_rounds):
        seed = RANDOM_SEEDS[round_idx]
        print(f"\n=== Round {round_idx} (seed={seed}) ===")
        
        results = run_regression(X_train, y_train, X_val, y_val, X_test, y_test, seed)
        all_results.append(results)
        
        for key, value in results.items():
            print(f"{key}: {value:.4f}")
    
    # Print average results
    print("\n=== Average Results ===")
    for key in all_results[0].keys():
        values = [r[key] for r in all_results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{key}: {mean_val:.4f} ± {std_val:.4f}")


if __name__ == "__main__":
    main()