"""
Downstream classification task for naive imputation methods.
Adapted for naive_imputation.h5 format (mean, median, locf, linear_interpolation).
"""

import argparse
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from global_config import RANDOM_SEEDS


class SimpleRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, n_classes=2):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, n_classes)
    
    def forward(self, x):
        _, h = self.rnn(x)
        h = torch.cat([h[-2], h[-1]], dim=-1)
        return self.fc(h)


class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, n_classes=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, n_classes)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)


def train_neural_classifier(model, train_loader, val_loader, epochs=50, lr=1e-3, device='cpu'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
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


def evaluate_neural_classifier(model, test_loader, device='cpu'):
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = torch.softmax(model(X_batch), dim=-1)
            all_probs.append(outputs[:, 1].cpu().numpy())
            all_labels.append(y_batch.numpy())
    
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    
    roc_auc = roc_auc_score(labels, probs)
    pr_auc = average_precision_score(labels, probs)
    
    return roc_auc, pr_auc


def run_classification(X_train, y_train, X_val, y_val, X_test, y_test, n_classes, seed):
    results = {}
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Flatten for XGBoost
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # Standardize
    scaler = StandardScaler()
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_val_flat = scaler.transform(X_val_flat)
    X_test_flat = scaler.transform(X_test_flat)
    
    # XGBoost
    xgb = XGBClassifier(n_estimators=100, max_depth=5, random_state=seed, eval_metric='logloss')
    xgb.fit(X_train_flat, y_train)
    probs = xgb.predict_proba(X_test_flat)[:, 1]
    results['XGB_ROC_AUC'] = roc_auc_score(y_test, probs)
    results['XGB_PR_AUC'] = average_precision_score(y_test, probs)
    
    # Prepare data for neural models
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=64)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=64)
    
    input_dim = X_train.shape[-1]
    
    # RNN
    rnn = SimpleRNN(input_dim, n_classes=n_classes)
    rnn = train_neural_classifier(rnn, train_loader, val_loader)
    results['RNN_ROC_AUC'], results['RNN_PR_AUC'] = evaluate_neural_classifier(rnn, test_loader)
    
    # Transformer
    transformer = SimpleTransformer(input_dim, n_classes=n_classes)
    transformer = train_neural_classifier(transformer, train_loader, val_loader)
    results['Transformer_ROC_AUC'], results['Transformer_PR_AUC'] = evaluate_neural_classifier(transformer, test_loader)
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_fold_path", type=str, required=True)
    parser.add_argument("--naive_method", type=str, required=True, 
                        choices=["mean", "median", "locf", "linear_interpolation"])
    parser.add_argument("--n_classes", type=int, default=2)
    parser.add_argument("--n_rounds", type=int, default=5)
    args = parser.parse_args()
    
    print(f"Running downstream classification for Naive_{args.naive_method}")
    print(f"Dataset: {args.dataset_fold_path}")
    
    # Load original data for labels
    with h5py.File(f"{args.dataset_fold_path}/train.h5", "r") as f:
        y_train = f["y"][:]
    with h5py.File(f"{args.dataset_fold_path}/val.h5", "r") as f:
        y_val = f["y"][:]
    with h5py.File(f"{args.dataset_fold_path}/test.h5", "r") as f:
        y_test = f["y"][:]
    
    # Load naive imputation results
    with h5py.File(f"{args.dataset_fold_path}/naive_imputation.h5", "r") as f:
        X_train = f[f"train/{args.naive_method}"][:]
        X_val = f[f"val/{args.naive_method}"][:]
        X_test = f[f"test/{args.naive_method}"][:]
    
    all_results = []
    for round_idx in range(args.n_rounds):
        seed = RANDOM_SEEDS[round_idx]
        print(f"\n=== Round {round_idx} (seed={seed}) ===")
        
        results = run_classification(X_train, y_train, X_val, y_val, X_test, y_test, args.n_classes, seed)
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