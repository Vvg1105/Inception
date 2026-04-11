"""
EEGNet for binary emotion decoding (sad / happy).

Architecture: Lawhern et al. 2018, adapted for short windows.

Input shape: (batch, 1, n_channels, n_timepoints)
  - n_channels   = 8  (BCI Core-8)
  - n_timepoints = 250 (1 s at 250 Hz)

Output: (batch, n_classes) logits
"""
import torch
import torch.nn as nn

EMOTIONS = ["sad", "happy"]

# ── Acquisition constants ─────────────────────────────────────────────────────
FS           = 250          # Hz
WINDOW_MS    = 1000         # target window in milliseconds
N_SAMPLES    = int(FS * WINDOW_MS / 1000)  # 250 samples
N_CHANNELS   = 8


class EEGNet(nn.Module):
    """
    EEGNet (compact CNN for EEG classification).

    Parameters
    ----------
    n_channels   : EEG electrode count
    n_timepoints : samples per window
    n_classes    : number of output classes
    F1           : number of temporal filters
    D            : depth multiplier for spatial (depthwise) conv
    F2           : number of separable conv output filters
    dropout      : dropout probability
    """

    def __init__(
        self,
        n_channels: int   = N_CHANNELS,
        n_timepoints: int = N_SAMPLES,
        n_classes: int    = len(EMOTIONS),
        F1: int  = 8,
        D: int   = 2,
        F2: int  = 16,
        dropout: float = 0.5,
    ):
        super().__init__()

        # ── Block 1: temporal filter → depthwise spatial filter ──────────────
        kern_t = max(3, n_timepoints // 4 * 2 + 1)   # odd, ~half-window
        pad_t  = kern_t // 2

        self.block1 = nn.Sequential(
            # Temporal convolution across all channels simultaneously
            nn.Conv2d(1, F1, kernel_size=(1, kern_t), padding=(0, pad_t), bias=False),
            nn.BatchNorm2d(F1),
            # Depthwise spatial convolution (one filter per channel)
            nn.Conv2d(F1, D * F1, kernel_size=(n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(D * F1),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )

        # ── Block 2: depthwise-separable temporal convolution ─────────────────
        self.block2 = nn.Sequential(
            # Depthwise
            nn.Conv2d(D * F1, D * F1, kernel_size=(1, 3), padding=(0, 1),
                      groups=D * F1, bias=False),
            # Pointwise
            nn.Conv2d(D * F1, F2, kernel_size=1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )

        # ── Classifier ────────────────────────────────────────────────────────
        # Dynamically compute flat feature size to stay robust to window length
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_timepoints)
            flat  = self.block2(self.block1(dummy)).numel()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        return self.classifier(x)


class EmotionMLP(nn.Module):
    """
    Lightweight MLP for emotion classification on small datasets.

    Extracts compact per-channel features from the raw window:
      - mean and std across time        → 2 * n_channels features
      - mean power in 4 EEG bands       → 4 * n_channels features
          delta 0.5–4 Hz, theta 4–8 Hz, alpha 8–13 Hz, beta 13–30 Hz

    Total input features: 6 * n_channels = 48  (for 8 channels)
    Total parameters    : ~1,700 — appropriate for datasets with <100 windows/class.

    Input shape : (batch, 1, n_channels, n_timepoints)  — same as EEGNet
    Output shape: (batch, n_classes) logits
    """

    # EEG frequency bands (Hz): delta, theta, alpha, beta
    BANDS = [(0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0)]

    def __init__(
        self,
        n_channels: int   = N_CHANNELS,
        n_timepoints: int = N_SAMPLES,
        n_classes: int    = len(EMOTIONS),
        fs: int           = FS,
        hidden: int       = 32,
        dropout: float    = 0.5,
    ):
        super().__init__()
        self.n_timepoints = n_timepoints
        self.fs           = fs
        feat = (2 + len(self.BANDS)) * n_channels   # mean, std + one per band
        self.net = nn.Sequential(
            nn.Linear(feat, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def _band_power(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute mean power in each EEG frequency band per channel.

        Parameters
        ----------
        x : (batch, n_channels, n_timepoints) — already squeezed

        Returns
        -------
        (batch, n_bands * n_channels)
        """
        N     = x.shape[2]
        freqs = torch.fft.rfftfreq(N, d=1.0 / self.fs, device=x.device)  # (N//2+1,)
        power = torch.fft.rfft(x, dim=2).abs().pow(2)                      # (B, C, N//2+1)
        parts = []
        for lo, hi in self.BANDS:
            mask = (freqs >= lo) & (freqs < hi)                             # (N//2+1,)
            parts.append(power[:, :, mask].mean(dim=2))                     # (B, C)
        return torch.cat(parts, dim=1)                                      # (B, n_bands*C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, n_channels, n_timepoints)
        x    = x.squeeze(1)                         # (batch, n_channels, n_timepoints)
        mean = x.mean(dim=2)                         # (batch, n_channels)
        std  = x.std(dim=2)                          # (batch, n_channels)
        bp   = self._band_power(x)                   # (batch, n_bands * n_channels)
        feat = torch.cat([mean, std, bp], dim=1)     # (batch, 6 * n_channels)
        return self.net(feat)


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = EEGNet()
    x     = torch.randn(4, 1, N_CHANNELS, N_SAMPLES)
    out   = model(x)
    print(f"Input : {tuple(x.shape)}")
    print(f"Output: {tuple(out.shape)}")
    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params:,}")
