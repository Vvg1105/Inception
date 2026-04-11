"""
EEGNet training on collected emotion data.

Loads eeg/data/eeg_raw.npy + eeg/data/eeg_labels.npy,
creates overlapping 100 ms windows, trains EEGNet, saves weights.

Uses temporal k-fold cross-validation: each class's recording is split into
K equal time blocks, and each fold holds out one block for validation while
training on the rest. This avoids the start-of-session vs end-of-session
distribution mismatch that occurs with a single fixed train/val boundary.

Output
------
  eeg/models/eegnet_emotion.pt   — model state dict (best fold)
  eeg/models/eegnet_config.json  — architecture / normalisation params

Run with conda base (has torch + sklearn + numpy):
  conda run -n base python eeg/train.py
"""
import os
import sys
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# ── Local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from eegnet import EEGNet, EmotionMLP, EMOTIONS, FS, N_SAMPLES, N_CHANNELS

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "models")
RAW_PATH   = os.path.join(DATA_DIR,  "eeg_raw.npy")
LABEL_PATH = os.path.join(DATA_DIR,  "eeg_labels.npy")
WEIGHTS    = os.path.join(MODEL_DIR, "eegnet_emotion.pt")
CFG_PATH   = os.path.join(MODEL_DIR, "eegnet_config.json")

# ── Model selection ───────────────────────────────────────────────────────────
# "mlp"    — EmotionMLP: ~500 params, good for small datasets (< ~100 windows/class)
# "eegnet" — EEGNet:   ~2500 params, better with larger datasets
MODEL = "mlp"

# ── Hyper-parameters ──────────────────────────────────────────────────────────
WINDOW       = N_SAMPLES   # 250 samples = 1 s
STRIDE       = 25          # 100 ms hop
TRAIN_RATIO  = 0.8         # 80 % of each class's samples → train, 20 % → val
BATCH        = 32
EPOCHS       = 150
LR           = 1e-3
WEIGHT_DECAY = 1e-3        # stronger regularisation for small dataset
PATIENCE     = 20          # early-stopping patience (per fold)
NOISE_STD    = 0.05        # additive Gaussian noise std (fraction of signal)
AMP_SCALE    = 0.2         # random amplitude scaling ±20 % — simulates session variability


# ── Windowing ─────────────────────────────────────────────────────────────────
def make_windows(raw: np.ndarray, labels: np.ndarray,
                 window: int, stride: int):
    """
    Slide a window of `window` samples with `stride` over the raw signal.
    Only emit windows where every sample has the same label (no cross-emotion
    boundary leakage).

    Returns
    -------
    X : float32 (N, 1, n_channels, window)
    y : int64   (N,)
    """
    X_list, y_list = [], []
    n = len(raw)
    for start in range(0, n - window + 1, stride):
        end      = start + window
        seg_lbl  = labels[start:end]
        if seg_lbl[0] != seg_lbl[-1]:
            continue                        # skip boundary windows
        if not np.all(seg_lbl == seg_lbl[0]):
            continue
        seg = raw[start:end]               # (window, n_channels)
        X_list.append(seg.T)               # (n_channels, window)
        y_list.append(seg_lbl[0])
    X = np.stack(X_list)[:, np.newaxis]    # (N, 1, n_channels, window)
    y = np.array(y_list, dtype=np.int64)
    return X.astype(np.float32), y


# ── 80/20 temporal split ──────────────────────────────────────────────────────
def split_train_val(raw_cls: np.ndarray, cls_idx: int,
                    train_ratio: float, window: int, stride: int):
    """
    Split raw_cls into train (first train_ratio) and val (remaining).
    A gap of `stride` samples at the boundary prevents window leakage.

    Returns (Xtr, ytr, Xva, yva).
    """
    n          = len(raw_cls)
    train_end  = int(n * train_ratio)
    val_start  = train_end + stride        # stride-sized gap prevents shared samples

    seg_train = raw_cls[:train_end]
    seg_val   = raw_cls[val_start:]

    lbl_train = np.full(len(seg_train), cls_idx, dtype=np.int64)
    lbl_val   = np.full(len(seg_val),   cls_idx, dtype=np.int64)

    Xtr, ytr = make_windows(seg_train, lbl_train, window, stride)
    Xva, yva = make_windows(seg_val,   lbl_val,   window, stride)
    return Xtr, ytr, Xva, yva


# ── Normalisation ─────────────────────────────────────────────────────────────
def normalise(X_train: np.ndarray, X_val: np.ndarray):
    """
    Z-score per channel across training windows.
    Returns scaled arrays + (mean, std) for saving.
    """
    N, _, C, T = X_train.shape
    flat        = X_train[:, 0].transpose(0, 2, 1).reshape(-1, C)  # (N*T, C)
    scaler      = StandardScaler()
    scaler.fit(flat)
    mean = scaler.mean_.astype(np.float32)
    std  = scaler.scale_.astype(np.float32)

    def apply(X):
        N, _, C, T = X.shape
        flat = X[:, 0].transpose(0, 2, 1).reshape(-1, C)
        flat = (flat - mean) / (std + 1e-8)
        return flat.reshape(N, T, C).transpose(0, 2, 1)[:, np.newaxis]

    return (apply(X_train).astype(np.float32),
            apply(X_val).astype(np.float32),
            mean, std)


# ── Build model ───────────────────────────────────────────────────────────────
def build_model(device):
    if MODEL == "mlp":
        m = EmotionMLP(n_channels=N_CHANNELS, n_timepoints=WINDOW,
                       n_classes=len(EMOTIONS))
    else:
        m = EEGNet(n_channels=N_CHANNELS, n_timepoints=WINDOW,
                   n_classes=len(EMOTIONS))
    return m.to(device)


# ── Training loop ─────────────────────────────────────────────────────────────
def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Load raw data ─────────────────────────────────────────────────────────
    if not os.path.exists(RAW_PATH):
        sys.exit(f"[ERROR] {RAW_PATH} not found — run collect_data.py first.")

    raw    = np.load(RAW_PATH)                 # (N, 8)
    labels = np.load(LABEL_PATH)               # (N,)
    print(f"Loaded {len(raw)} samples, {len(EMOTIONS)} classes")
    for i, e in enumerate(EMOTIONS):
        print(f"  {e:>8s}: {(labels==i).sum()} raw samples")

    # ── 80/20 temporal split per class ───────────────────────────────────────
    print(f"\nWindowing  (window={WINDOW} samples = {WINDOW/FS*1000:.0f} ms,"
          f"  stride={STRIDE} samples = {STRIDE/FS*1000:.0f} ms)")
    print(f"  split: first {int(TRAIN_RATIO*100)}% train / last {int((1-TRAIN_RATIO)*100)}% val\n")

    X_tr_parts, y_tr_parts = [], []
    X_va_parts, y_va_parts = [], []
    for cls_idx, cls_name in enumerate(EMOTIONS):
        raw_cls = raw[labels == cls_idx]
        Xtr, ytr, Xva, yva = split_train_val(
            raw_cls, cls_idx, TRAIN_RATIO, WINDOW, STRIDE)
        X_tr_parts.append(Xtr); y_tr_parts.append(ytr)
        X_va_parts.append(Xva); y_va_parts.append(yva)
        print(f"  {cls_name:>8s}: {len(Xtr):3d} train  |  {len(Xva):3d} val")

    X_train = np.concatenate(X_tr_parts); y_train = np.concatenate(y_tr_parts)
    X_val   = np.concatenate(X_va_parts); y_val   = np.concatenate(y_va_parts)
    print(f"\n  total: {len(X_train)} train  |  {len(X_val)} val")

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    X_train, X_val, ch_mean, ch_std = normalise(X_train, X_val)

    def to_tensor(X, y):
        return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))

    train_loader = DataLoader(to_tensor(X_train, y_train),
                              batch_size=BATCH, shuffle=True, drop_last=False)
    val_loader   = DataLoader(to_tensor(X_val, y_val),
                              batch_size=BATCH, shuffle=False, drop_last=False)

    model     = build_model(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {MODEL}  |  Parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=7, factor=0.5, min_lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    best_val_acc  = 0.0
    best_val_loss = float("inf")
    best_epoch    = 0
    no_improve    = 0

    print(f"\n{'Epoch':>6}  {'TrainLoss':>10}  {'TrainAcc':>9}  "
          f"{'ValLoss':>8}  {'ValAcc':>7}  {'LR':>8}")
    print("─" * 64)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            xb = xb + NOISE_STD * torch.randn_like(xb)
            scale = 1.0 + AMP_SCALE * (2 * torch.rand(xb.shape[0], 1, 1, 1,
                                                       device=device) - 1)
            xb = xb * scale
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
                va_loss += criterion(logits, yb).item() * len(yb)
                va_correct += (logits.argmax(1) == yb).sum().item()
                va_total   += len(yb)

        tr_acc  = tr_correct / tr_total
        va_acc  = va_correct / va_total
        tr_loss /= tr_total
        va_loss /= va_total
        scheduler.step(va_loss)
        cur_lr = optimizer.param_groups[0]["lr"]

        marker = " *" if va_acc > best_val_acc else ""
        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_acc:>9.4f}  "
              f"{va_loss:>8.4f}  {va_acc:>7.4f}  {cur_lr:>8.6f}{marker}")

        if va_acc > best_val_acc:
            best_val_acc  = va_acc
            best_val_loss = va_loss
            best_epoch    = epoch
            no_improve    = 0
            torch.save(model.state_dict(), WEIGHTS)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(best val acc: {best_val_acc:.4f} @ epoch {best_epoch})")
                break

    print(f"\nBest val acc: {best_val_acc:.4f}  (epoch {best_epoch})")

    model.load_state_dict(torch.load(WEIGHTS, map_location=device, weights_only=True))
    model.eval()
    X_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)
    with torch.no_grad():
        preds = model(X_val_t).argmax(dim=1)
    print("\nPer-class val accuracy:")
    for i, name in enumerate(EMOTIONS):
        mask    = y_val_t == i
        correct = (preds[mask] == i).sum().item()
        total   = mask.sum().item()
        bar     = "█" * int(correct / total * 20) if total else ""
        print(f"  {name:>8s}: {correct:3d}/{total:3d}  {correct/total*100:5.1f}%  {bar}"
              if total else f"  {name:>8s}: no val samples")

    print(f"\nWeights saved: {WEIGHTS}")

    # ── Save config for live_decode ───────────────────────────────────────────
    config = {
        "model":        MODEL,
        "n_channels":   N_CHANNELS,
        "n_timepoints": WINDOW,
        "n_classes":    len(EMOTIONS),
        "emotions":     EMOTIONS,
        "fs":           FS,
        "ch_mean":      ch_mean.tolist(),
        "ch_std":       ch_std.tolist(),
    }
    with open(CFG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved:  {CFG_PATH}\n")


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    train()
