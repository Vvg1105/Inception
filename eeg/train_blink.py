"""
Blink classifier training.

Loads eeg/data/blink_raw.npy + eeg/data/blink_labels.npy,
creates overlapping 500 ms windows, trains EEGNet (binary), saves weights.

Output
------
  eeg/models/blinknet.pt          — model state dict
  eeg/models/blinknet_config.json — architecture / normalisation params

Run:
  conda run -n base python eeg/train_blink.py
"""
import os
import sys
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from eegnet import EEGNet

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR   = os.path.join(os.path.dirname(__file__), "models")
RAW_PATH    = os.path.join(DATA_DIR,  "blink_raw.npy")
LABEL_PATH  = os.path.join(DATA_DIR,  "blink_labels.npy")
WEIGHTS     = os.path.join(MODEL_DIR, "blinknet.pt")
CFG_PATH    = os.path.join(MODEL_DIR, "blinknet_config.json")

CLASSES = ["blink", "open"]   # label 0 = blink, 1 = open

# ── Hyper-parameters ───────────────────────────────────────────────────────────
FS           = 250
WINDOW_MS    = 500             # 500 ms — captures a full blink waveform
WINDOW       = int(FS * WINDOW_MS / 1000)   # 125 samples
STRIDE       = 5          # 20 ms hop → many windows; temporal split prevents leakage
N_CHANNELS   = 8

TRAIN_SECS   = 15         # first 15 s of each class → train windows
VAL_SECS     = 5          # last  5 s of each class  → val  windows
BATCH        = 64
EPOCHS       = 150
LR           = 1e-3
WEIGHT_DECAY = 1e-3       # stronger regularisation for small dataset
PATIENCE     = 20
NOISE_STD    = 0.05       # Gaussian noise std added to train batches


# ── Windowing ──────────────────────────────────────────────────────────────────
def make_windows(raw: np.ndarray, labels: np.ndarray, window: int, stride: int):
    """
    Slide a window over the raw signal.  Only emit windows where every
    sample shares the same label (no blink/open boundary leakage).

    Returns
    -------
    X : float32 (N, 1, n_channels, window)
    y : int64   (N,)
    """
    X_list, y_list = [], []
    for start in range(0, len(raw) - window + 1, stride):
        seg_lbl = labels[start:start + window]
        if not np.all(seg_lbl == seg_lbl[0]):
            continue
        seg = raw[start:start + window]    # (window, n_channels)
        X_list.append(seg.T)              # (n_channels, window)
        y_list.append(int(seg_lbl[0]))
    X = np.stack(X_list)[:, np.newaxis]   # (N, 1, n_channels, window)
    return X.astype(np.float32), np.array(y_list, dtype=np.int64)


# ── Normalisation ──────────────────────────────────────────────────────────────
def normalise(X_train, X_val):
    N, _, C, T = X_train.shape
    flat        = X_train[:, 0].transpose(0, 2, 1).reshape(-1, C)
    scaler      = StandardScaler().fit(flat)
    mean        = scaler.mean_.astype(np.float32)
    std         = scaler.scale_.astype(np.float32)

    def apply(X):
        N, _, C, T = X.shape
        f = X[:, 0].transpose(0, 2, 1).reshape(-1, C)
        f = (f - mean) / (std + 1e-8)
        return f.reshape(N, T, C).transpose(0, 2, 1)[:, np.newaxis].astype(np.float32)

    return apply(X_train), apply(X_val), mean, std


# ── Train ──────────────────────────────────────────────────────────────────────
def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(RAW_PATH):
        sys.exit(f"[ERROR] {RAW_PATH} not found — run collect_blink_data.py first.")

    raw    = np.load(RAW_PATH)    # (N, 8)
    labels = np.load(LABEL_PATH)  # (N,)
    print(f"Loaded {len(raw)} samples")
    for name, i in enumerate(CLASSES):
        print(f"  {i:>6}: {(labels == name).sum()} raw samples")

    # ── Window each class from separate raw segments (zero leakage) ──────────
    n_train_raw = TRAIN_SECS * FS
    n_val_raw   = VAL_SECS   * FS

    X_tr_parts, y_tr_parts = [], []
    X_va_parts, y_va_parts = [], []

    print(f"\nWindowing  (window={WINDOW_MS}ms, stride={STRIDE} samples = {STRIDE/FS*1000:.0f}ms)")
    print(f"  train: first {TRAIN_SECS}s per class  |  val: last {VAL_SECS}s per class\n")

    for cls_idx, cls_name in enumerate(CLASSES):
        raw_cls  = raw[labels == cls_idx]
        seg_tr   = raw_cls[:n_train_raw]
        seg_va   = raw_cls[len(raw_cls) - n_val_raw:]
        lbl_tr   = np.full(len(seg_tr), cls_idx, dtype=np.int64)
        lbl_va   = np.full(len(seg_va), cls_idx, dtype=np.int64)
        Xtr, ytr = make_windows(seg_tr, lbl_tr, WINDOW, STRIDE)
        Xva, yva = make_windows(seg_va, lbl_va, WINDOW, STRIDE)
        X_tr_parts.append(Xtr); y_tr_parts.append(ytr)
        X_va_parts.append(Xva); y_va_parts.append(yva)
        print(f"  {cls_name:>6}: {len(Xtr):4d} train windows  |  {len(Xva):4d} val windows")

    X_train = np.concatenate(X_tr_parts); y_train = np.concatenate(y_tr_parts)
    X_val   = np.concatenate(X_va_parts); y_val   = np.concatenate(y_va_parts)
    print(f"\n  total: {len(X_train)} train  |  {len(X_val)} val")

    X_train, X_val, ch_mean, ch_std = normalise(X_train, X_val)

    device = torch.device("mps"  if torch.backends.mps.is_available()  else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    def ds(X, y):
        return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))

    train_loader = DataLoader(ds(X_train, y_train), batch_size=BATCH, shuffle=True)
    val_loader   = DataLoader(ds(X_val,   y_val),   batch_size=BATCH, shuffle=False)

    # EEGNet reused as a binary classifier
    model = EEGNet(n_channels=N_CHANNELS, n_timepoints=WINDOW,
                   n_classes=len(CLASSES)).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=6, factor=0.5, min_lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_epoch, no_improve = 0, 0

    print(f"\n{'Epoch':>6}  {'TrLoss':>8}  {'TrAcc':>7}  {'VaLoss':>8}  {'VaAcc':>7}")
    print("─" * 50)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            xb = xb + NOISE_STD * torch.randn_like(xb)
            optimizer.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            tr_loss    += loss.item() * len(yb)
            tr_correct += (logits.argmax(1) == yb).sum().item()
            tr_total   += len(yb)

        model.eval()
        va_loss, va_correct, va_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits  = model(xb)
                va_loss    += criterion(logits, yb).item() * len(yb)
                va_correct += (logits.argmax(1) == yb).sum().item()
                va_total   += len(yb)

        tr_acc   = tr_correct / tr_total
        va_acc   = va_correct / va_total
        va_loss /= va_total
        scheduler.step(va_loss)

        marker = " *" if va_loss < best_val_loss else ""
        print(f"{epoch:>6}  {tr_loss/tr_total:>8.4f}  {tr_acc:>7.4f}  "
              f"{va_loss:>8.4f}  {va_acc:>7.4f}{marker}")

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_epoch    = epoch
            no_improve    = 0
            torch.save(model.state_dict(), WEIGHTS)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\nEarly stop at epoch {epoch}  "
                      f"(best val loss: {best_val_loss:.4f} @ epoch {best_epoch})")
                break

    print(f"\nBest val loss: {best_val_loss:.4f}  (epoch {best_epoch})")
    print(f"Weights: {WEIGHTS}")

    config = {
        "n_channels":   N_CHANNELS,
        "n_timepoints": WINDOW,
        "n_classes":    len(CLASSES),
        "classes":      CLASSES,
        "fs":           FS,
        "window_ms":    WINDOW_MS,
        "ch_mean":      ch_mean.tolist(),
        "ch_std":       ch_std.tolist(),
    }
    with open(CFG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config: {CFG_PATH}\n")


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    train()
