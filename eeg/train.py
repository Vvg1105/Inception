"""
EEGNet training on collected emotion data.

Loads all run_NNN_raw/labels.npy files from eeg/data/.

Split strategy
--------------
  • 2+ runs  →  earlier runs = train, last run = val.
                No window ever spans a run boundary, so there is zero leakage.
  • 1 run    →  80/20 temporal split within that run (per class), with a
                stride-wide gap at the boundary to prevent shared samples.

Output
------
  With --gtec  : eeg/models/eegnet_emotion_gtec.pt,  eegnet_config_gtec.json
  With --cyton : eeg/models/eegnet_emotion_cyton.pt, eegnet_config_cyton.json

Binary vs multi-class
----------------------
  Set TRAIN_EMOTIONS in this file to a list of two names from eeg/eegnet.py EMOTIONS
  (e.g. ["sad", "happy"]) to train a 2-way classifier; samples with other labels are
  dropped.  Use TRAIN_EMOTIONS = None for all classes in EMOTIONS.

Run (pick one board; data must live under eeg/data/<board>/ from collect_data.py):
  python eeg/train.py --cyton
  python eeg/train.py --gtec
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
WEIGHTS    = os.path.join(MODEL_DIR, "eegnet_emotion.pt")
CFG_PATH   = os.path.join(MODEL_DIR, "eegnet_config.json")

# ── Model selection ───────────────────────────────────────────────────────────
# "mlp"    — EmotionMLP: band-power features, more cross-session stable (recommended)
# "eegnet" — EEGNet:     raw waveform CNN, needs more data to generalise across sessions
MODEL = "eegnet"

# ── Label selection ───────────────────────────────────────────────────────────
# Subset of EMOTIONS to train on. Set to None to use all emotions.
# Examples:
#   TRAIN_EMOTIONS = ["sad", "happy"]      # binary classifier, ignores neutral data
#   TRAIN_EMOTIONS = ["happy", "neutral"]  # binary, ignores sad data
#   TRAIN_EMOTIONS = None                  # use all emotions (default)
TRAIN_EMOTIONS = ["sad", "happy"]

# ── Hyper-parameters ──────────────────────────────────────────────────────────
WINDOW       = N_SAMPLES   # 250 samples = 1 s
STRIDE       = 25          # 100 ms hop
TRAIN_RATIO  = 0.8         # 80 % of each class's samples → train, 20 % → val
BATCH        = 32
EPOCHS       = 150
LR           = 1e-3
WEIGHT_DECAY = 3e-3        # stronger regularisation (was 1e-3)
PATIENCE     = 20          # early-stopping patience (per fold)
NOISE_STD    = 0.15        # additive Gaussian noise — simulate session variability (was 0.05)
AMP_SCALE    = 0.4         # random amplitude scaling ±40 % (was 0.2)


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
def build_model(device, n_classes):
    if MODEL == "mlp":
        m = EmotionMLP(n_channels=N_CHANNELS, n_timepoints=WINDOW,
                       n_classes=n_classes)
    else:
        m = EEGNet(n_channels=N_CHANNELS, n_timepoints=WINDOW,
                   n_classes=n_classes)
    return m.to(device)


# ── Training loop ─────────────────────────────────────────────────────────────
def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Discover runs ─────────────────────────────────────────────────────────
    import re
    run_nums = sorted(
        int(m.group(1))
        for f in os.listdir(DATA_DIR)
        if (m := re.match(r"run_(\d+)_raw\.npy$", f))
        and os.path.exists(os.path.join(DATA_DIR, f"run_{int(m.group(1)):03d}_labels.npy"))
    )
    if not run_nums:
        sys.exit("[ERROR] No run_NNN_raw.npy files found — run collect_data.py first.")

    print(f"Found {len(run_nums)} run(s): {[f'run_{r:03d}' for r in run_nums]}")

    # ── Resolve active emotion set ────────────────────────────────────────────
    active_emotions = TRAIN_EMOTIONS if TRAIN_EMOTIONS is not None else EMOTIONS
    for e in active_emotions:
        if e not in EMOTIONS:
            sys.exit(f"[ERROR] '{e}' not in EMOTIONS {EMOTIONS} — check TRAIN_EMOTIONS.")
    if TRAIN_EMOTIONS is not None:
        print(f"Training on subset: {active_emotions}")

    def load_run(n):
        raw       = np.load(os.path.join(DATA_DIR, f"run_{n:03d}_raw.npy"))
        labels_raw = np.load(os.path.join(DATA_DIR, f"run_{n:03d}_labels.npy"))
        meta_path = os.path.join(DATA_DIR, f"run_{n:03d}_meta.json")

        # Read per-run emotion list so index 2 in an old run (neutral) is not
        # confused with index 2 in a new run (angry) — they have different metas.
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                run_emotions = json.load(f)["emotions"]
        else:
            # Legacy runs with no meta: assume they used the current EMOTIONS list
            run_emotions = EMOTIONS
            print(f"  [warn] run_{n:03d} has no meta file — assuming emotions={EMOTIONS}")

        # Build a mapping from this run's label index → active_emotions index.
        # Only keep samples whose emotion name appears in active_emotions.
        keep_map = {}  # run label index → new label index
        for run_idx, emo_name in enumerate(run_emotions):
            if emo_name in active_emotions:
                keep_map[run_idx] = active_emotions.index(emo_name)

        mask   = np.array([l in keep_map for l in labels_raw])
        raw    = raw[mask]
        labels = np.array([keep_map[l] for l in labels_raw[mask]], dtype=np.int64)
        return raw, labels

    print(f"\nWindowing  (window={WINDOW} samples = {WINDOW/FS*1000:.0f} ms,"
          f"  stride={STRIDE} samples = {STRIDE/FS*1000:.0f} ms)")

    X_tr_parts, y_tr_parts = [], []
    X_va_parts, y_va_parts = [], []

    if len(run_nums) == 1:
        # Single run: 80/20 temporal split per class
        print(f"  Single run → 80/20 temporal split per class "
              f"(first {int(TRAIN_RATIO*100)}% train / last {int((1-TRAIN_RATIO)*100)}% val)\n")
        raw, labels = load_run(run_nums[0])
        print(f"  run_001: {len(raw)} samples")
        for i, e in enumerate(active_emotions):
            print(f"    {e:>8s}: {(labels==i).sum()} raw samples")
        for cls_idx, cls_name in enumerate(active_emotions):
            raw_cls = raw[labels == cls_idx]
            Xtr, ytr, Xva, yva = split_train_val(
                raw_cls, cls_idx, TRAIN_RATIO, WINDOW, STRIDE)
            X_tr_parts.append(Xtr); y_tr_parts.append(ytr)
            X_va_parts.append(Xva); y_va_parts.append(yva)
            print(f"  {cls_name:>8s}: {len(Xtr):3d} train  |  {len(Xva):3d} val")
    else:
        # Multiple runs: earlier runs = train, last run = val
        train_runs = run_nums[:-1]
        val_run    = run_nums[-1]
        print(f"  Train runs: {[f'run_{r:03d}' for r in train_runs]}")
        print(f"  Val   run : run_{val_run:03d}\n")

        for split_label, runs, tr_parts, va_parts in [
            ("train", train_runs, (X_tr_parts, y_tr_parts), None),
            ("val",   [val_run],  None, (X_va_parts, y_va_parts)),
        ]:
            target_X = tr_parts[0] if tr_parts else va_parts[0]
            target_y = tr_parts[1] if tr_parts else va_parts[1]
            for run_n in runs:
                raw, labels = load_run(run_n)
                print(f"  run_{run_n:03d} ({split_label}): {len(raw)} samples", end="")
                for i, e in enumerate(active_emotions):
                    print(f"  {e}={int((labels==i).sum())}", end="")
                print()
                for cls_idx in range(len(active_emotions)):
                    raw_cls = raw[labels == cls_idx]
                    lbl_arr = np.full(len(raw_cls), cls_idx, dtype=np.int64)
                    X, y    = make_windows(raw_cls, lbl_arr, WINDOW, STRIDE)
                    target_X.append(X)
                    target_y.append(y)

        print(f"\n  train windows per class:")
        for cls_idx, cls_name in enumerate(active_emotions):
            n = sum((y == cls_idx).sum() for y in y_tr_parts)
            print(f"    {cls_name:>8s}: {n}")
        print(f"  val windows per class:")
        for cls_idx, cls_name in enumerate(active_emotions):
            n = sum((y == cls_idx).sum() for y in y_va_parts)
            print(f"    {cls_name:>8s}: {n}")

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

    model     = build_model(device, n_classes=len(active_emotions))
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
    for i, name in enumerate(active_emotions):
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
        "n_classes":    len(active_emotions),
        "emotions":     active_emotions,
        "fs":           FS,
        "ch_mean":      ch_mean.tolist(),
        "ch_std":       ch_std.tolist(),
    }
    with open(CFG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved:  {CFG_PATH}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="EEGNet emotion training")
    board_group = parser.add_mutually_exclusive_group(required=True)
    board_group.add_argument("--gtec",  action="store_true", help="Train on g.tec data (eeg/data/gtec/)")
    board_group.add_argument("--cyton", action="store_true", help="Train on Cyton data  (eeg/data/cyton/)")
    args = parser.parse_args()

    board = "gtec" if args.gtec else "cyton"
    # Rebind module-level paths before train() (same names as lines 36–39).
    DATA_DIR = os.path.join(os.path.dirname(__file__), "data", board)
    WEIGHTS  = os.path.join(MODEL_DIR, f"eegnet_emotion_{board}.pt")
    CFG_PATH = os.path.join(MODEL_DIR, f"eegnet_config_{board}.json")

    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Board : {board.upper()}  |  data: {DATA_DIR}")

    np.random.seed(42)
    torch.manual_seed(42)
    train()
