"""Simple time-domain features for short EMG windows."""

from __future__ import annotations

import numpy as np


def emg_features(window: np.ndarray) -> np.ndarray:
    """
    One row of features per window (1D array of length n_features).
    Uses mean-centered signal for ZC / MAV-style stats.
    """
    x = np.asarray(window, dtype=np.float64).ravel()
    if x.size == 0:
        return np.zeros(8, dtype=np.float64)
    mu = float(np.mean(x))
    xc = x - mu
    rms = float(np.sqrt(np.mean(xc**2)))
    mav = float(np.mean(np.abs(xc)))
    std = float(np.std(xc))
    if xc.size < 2:
        zc = 0
    else:
        zc = int(np.sum(xc[:-1] * xc[1:] < 0))
    return np.array(
        [
            mu,
            std,
            rms,
            mav,
            float(np.min(x)),
            float(np.max(x)),
            float(np.ptp(x)),
            float(zc),
        ],
        dtype=np.float64,
    )


def featurize_dataset(X: np.ndarray) -> np.ndarray:
    """X: (n_trials, n_samples) -> (n_trials, n_features)"""
    return np.stack([emg_features(X[i]) for i in range(X.shape[0])], axis=0)
