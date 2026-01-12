"""
Downstream forecasting task for naive imputation methods.
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


class ForecastRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, forecast_horizon=5):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, forecast_horizon)
    
    def forward(self, x):
        _, h = self.rnn(x)
        h = torch.cat([h[-2], h[-1]], dim=-1)
        return self.fc(h)


class ForecastTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, forecast_horizon=5):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, forecast_horizon)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)


def train_neural_forecaster(model, train_loader, val_loader, epochs=50, lr=1e-3, device='cpu'):
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


def evaluate_neural_forecaster(model, test_loader, device='cpu'):
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


def run_forecasting(X_train, y_train, X_val, y_val, X_test, y_test, seed, forecast_horizon=5):
    results = {}
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Flatten for XGBoost
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # Standardize
    scaler_X = StandardScaler()
    X_train_flat = scaler_X.fit_transform(X_train_flat)
    X_val_flat = scaler_X.transform(X_val_flat)
    X_test_flat = scaler_X.transform(X_test_flat)
    
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)
    y_test_scaled = scaler_y.transform(y_test)
    
    # XGBoost (multi-output)
    preds_xgb = np.zeros_like(y_test)
    for i in range(forecast_horizon):
        xgb = XGBRegressor(n_estimators=100, max_depth=5, random_state=seed)
        xgb.fit(X_train_flat, y_train_scaled[:, i])
        preds_xgb[:, i] = xgb.predict(X_test_flat)
    
    preds_xgb = scaler_y.inverse_transform(preds_xgb)
    results['XGB_MAE'] = mean_absolute_error(y_test.flatten(), preds_xgb.flatten())
    results['XGB_MSE'] = mean_squared_error(y_test.flatten(), preds_xgb.flatten())
    results['XGB_MRE'] = np.mean(np.abs(y_test - preds_xgb) / (np.abs(y_test) + 1e-8))
    
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
    rnn = ForecastRNN(input_dim, forecast_horizon=forecast_horizon)
    rnn = train_neural_forecaster(rnn, train_loader, val_loader)
    results['RNN_MAE'], results['RNN_MSE'], results['RNN_MRE'] = evaluate_neural_forecaster(rnn, test_loader)
    
    # Transformer
    transformer = ForecastTransformer(input_dim, forecast_horizon=forecast_horizon)
    transformer = train_neural_forecaster(transformer, train_loader, val_loader)
    results['Transformer_MAE'], results['Transformer_MSE'], results['Transformer_MRE'] = evaluate_neural_forecaster(transformer, test_loader)
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_fold_path", type=str, required=True)
    parser.add_argument("--naive_method", type=str, required=True, 
                        choices=["mean", "median", "locf", "linear_interpolation"])
    parser.add_argument("--forecast_horizon", type=int, default=5)
    parser.add_argument("--n_rounds", type=int, default=5)
    args = parser.parse_args()
    
    print(f"Running downstream forecasting for Naive_{args.naive_method}")
    print(f"Dataset: {args.dataset_fold_path}")
    
    # Load original data for targets
    with h5py.File(f"{args.dataset_fold_path}/train.h5", "r") as f:
        X_ori_train = f["X_ori"][:]
    with h5py.File(f"{args.dataset_fold_path}/val.h5", "r") as f:
        X_ori_val = f["X_ori"][:]
    with h5py.File(f"{args.dataset_fold_path}/test.h5", "r") as f:
        X_ori_test = f["X_ori"][:]
    
    # Target: last feature, last forecast_horizon steps
    y_train = X_ori_train[:, -args.forecast_horizon:, -1]
    y_val = X_ori_val[:, -args.forecast_horizon:, -1]
    y_test = X_ori_test[:, -args.forecast_horizon:, -1]
    
    # Load naive imputation results (excluding last forecast_horizon steps)
    with h5py.File(f"{args.dataset_fold_path}/naive_imputation.h5", "r") as f:
        X_train = f[f"train/{args.naive_method}"][:][:, :-args.forecast_horizon, :]
        X_val = f[f"val/{args.naive_method}"][:][:, :-args.forecast_horizon, :]
        X_test = f[f"test/{args.naive_method}"][:][:, :-args.forecast_horizon, :]
    
    all_results = []
    for round_idx in range(args.n_rounds):
        seed = RANDOM_SEEDS[round_idx]
        print(f"\n=== Round {round_idx} (seed={seed}) ===")
        
        results = run_forecasting(X_train, y_train, X_val, y_val, X_test, y_test, seed, args.forecast_horizon)
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