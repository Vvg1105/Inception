"""
EEGNet for 4-class emotion decoding.

Architecture: Lawhern et al. 2018, adapted for short windows.

Input shape: (batch, 1, n_channels, n_timepoints)
  - n_channels   = 8  (BCI Core-8)
  - n_timepoints = 25 (~100 ms at 250 Hz)

Output: (batch, n_classes) logits
"""
import torch
import torch.nn as nn

EMOTIONS = ["angry", "sad", "happy", "fear"]

# ── Acquisition constants ─────────────────────────────────────────────────────
FS           = 250          # Hz
WINDOW_MS    = 100          # target window in milliseconds
N_SAMPLES    = int(FS * WINDOW_MS / 1000)  # 25 samples
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


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = EEGNet()
    x     = torch.randn(4, 1, N_CHANNELS, N_SAMPLES)
    out   = model(x)
    print(f"Input : {tuple(x.shape)}")
    print(f"Output: {tuple(out.shape)}")
    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params:,}")
