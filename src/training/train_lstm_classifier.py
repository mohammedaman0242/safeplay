#!/usr/bin/env python3
"""
train_lstm_classifier.py

Train a supervised LSTM classifier to distinguish:
- SAFE windows (label 0)
- UNSAFE windows (label 1)

Uses:
  SAFE:   final_safe/all_actions_safe.npz  (key: data)
  UNSAFE: windows/windows_unsafe/<action>_windows_norm.npz

All windows must be normalized with same feature set & shape (T,F).
"""

import os
import numpy as np
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# ---------------- CONFIG ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SAFE_FILE = os.path.join(BASE_DIR, "final_safe", "all_actions_safe.npz")
UNSAFE_FILES = [
    os.path.join(BASE_DIR, "windows", "windows_unsafe", "attack_windows_norm.npz"),
    os.path.join(BASE_DIR, "windows", "windows_unsafe", "block_windows_norm.npz"),
    os.path.join(BASE_DIR, "windows", "windows_unsafe", "defence_windows_norm.npz"),
]

MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# ---------------- DATASET ----------------
class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype("float32")
        self.y = y.astype("int64")

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]      # (T,F)
        y = self.y[idx]      # scalar
        return torch.from_numpy(x), torch.tensor(y)


# ---------------- MODEL ----------------
class LSTMClassifier(nn.Module):
    def __init__(self, feat_dim, hidden_dim=128, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (B,T,F)
        _, (h_n, _) = self.lstm(x)         # h_n: (num_layers,B,H)
        h_last = h_n[-1]                   # (B,H)
        logits = self.fc(h_last)           # (B,1)
        return logits.squeeze(1)           # (B,)


# ---------------- HELPERS ----------------
def load_safe_unsafe():
    # SAFE
    d_safe = np.load(SAFE_FILE)
    X_safe = d_safe["data"].astype("float32")   # (N_s, T, F)
    N_s, T, F = X_safe.shape
    print("Loaded SAFE:", SAFE_FILE, "shape:", X_safe.shape)

    # UNSAFE
    X_unsafe_list = []
    for f in UNSAFE_FILES:
        if not os.path.exists(f):
            print("WARNING: missing unsafe file:", f)
            continue
        d = np.load(f)
        X_u = d["data"].astype("float32")
        print("Loaded UNSAFE:", f, "shape:", X_u.shape)
        X_unsafe_list.append(X_u)

    if not X_unsafe_list:
        raise RuntimeError("No unsafe files loaded!")

    X_unsafe = np.concatenate(X_unsafe_list, axis=0)   # (N_u, T, F)
    print("Combined UNSAFE shape:", X_unsafe.shape)

    # Check same T,F
    if X_unsafe.shape[1] != T or X_unsafe.shape[2] != F:
        raise ValueError("SAFE and UNSAFE shapes mismatch in T or F")

    # Labels
    y_safe = np.zeros((X_safe.shape[0],), dtype=np.int64)
    y_unsafe = np.ones((X_unsafe.shape[0],), dtype=np.int64)

    # Concatenate
    X = np.concatenate([X_safe, X_unsafe], axis=0)
    y = np.concatenate([y_safe, y_unsafe], axis=0)

    print("TOTAL dataset:", X.shape, "labels:", y.shape,
          "| unsafe ratio:", y_unsafe.shape[0] / y.shape[0])
    return X, y, T, F


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    losses = []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE).float()

        optimizer.zero_grad()
        logits = model(X_batch)           # (B,)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0


def eval_epoch(model, loader, criterion):
    model.eval()
    losses = []
    all_y = []
    all_pred = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE).float()

            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            losses.append(loss.item())

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()     # threshold 0.5
            all_y.append(y_batch.cpu().numpy())
            all_pred.append(preds.cpu().numpy())

    if not losses:
        return 0.0, 0, 0, 0, 0, 0.0, 0.0

    y_true = np.concatenate(all_y).astype("int32")
    y_pred = np.concatenate(all_pred).astype("int32")

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    return float(np.mean(losses)), acc, prec, rec, f1


# ---------------- MAIN ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    args = parser.parse_args()

    print("=== LSTM Classifier Training (SAFE vs UNSAFE) ===")

    # 1) Load data
    X, y, T, F = load_safe_unsafe()

    # 2) Train/val split (stratified)
    idx = np.arange(X.shape[0])
    from sklearn.model_selection import train_test_split
    train_idx, val_idx = train_test_split(
        idx, test_size=args.val_ratio, shuffle=True,
        stratify=y, random_state=42
    )

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    print("Train size:", X_train.shape[0], "Val size:", X_val.shape[0])

    train_ds = SequenceDataset(X_train, y_train)
    val_ds = SequenceDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, drop_last=False)

    # 3) Model, optimizer, loss
    model = LSTMClassifier(feat_dim=F, hidden_dim=args.hidden, num_layers=args.layers).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_f1 = 0.0
    best_path = os.path.join(MODEL_DIR, "lstm_classifier_best.pth")

    # 4) Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, acc, prec, rec, f1 = eval_epoch(model, val_loader, criterion)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"acc={acc:.3f} | prec={prec:.3f} | rec={rec:.3f} | f1={f1:.3f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), best_path)

    print("Training done. Best F1:", best_f1)
    print("Best model saved to:", best_path)


if __name__ == "__main__":
    main()
