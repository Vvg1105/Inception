"""
Render TRIBE v2 pooled vertex activations onto an fsaverage5 pial brain surface using nilearn.

TRIBE v2 outputs shape ``(n_timepoints, 29286)`` — 29,286 vertices total:
  - First 20,484 are cortical (fsaverage5): ``[:10242]`` = left hemi, ``[10242:20484]`` = right hemi.
  - Remaining 8,802 are subcortical and not plotted on the surface.

Returns a PNG as ``bytes``.
"""
from __future__ import annotations

import base64
import io
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

CORTICAL_VERTICES = 20_484
HEMI_VERTICES = 10_242


def render_tribe_brain_png(
    pooled: np.ndarray,
    *,
    dpi: int = 100,
    figsize: tuple[float, float] = (5, 3.5),
) -> bytes:
    """Render the TRIBE pooled vector onto fsaverage5 and return PNG bytes.

    Parameters
    ----------
    pooled : 1-D array of length >= 20484 (full TRIBE output is 29286).
    dpi, figsize : resolution controls.
    """
    from nilearn import datasets, plotting  # heavy imports, keep lazy

    pooled = np.asarray(pooled, dtype=np.float64).ravel()
    n = pooled.shape[0]
    if n < CORTICAL_VERTICES:
        raise ValueError(
            f"pooled vector has {n} values, need at least {CORTICAL_VERTICES} cortical vertices"
        )

    cortical = pooled[:CORTICAL_VERTICES]
    lh = cortical[:HEMI_VERTICES]
    rh = cortical[HEMI_VERTICES:CORTICAL_VERTICES]

    fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")

    fig, axes = plt.subplots(1, 2, figsize=figsize, subplot_kw={"projection": "3d"})

    for ax, (hemi_data, hemi_mesh, hemi_bg, hemi_label) in zip(
        axes,
        [
            (lh, fsaverage["pial_left"], fsaverage["sulc_left"], "left"),
            (rh, fsaverage["pial_right"], fsaverage["sulc_right"], "right"),
        ],
    ):
        plotting.plot_surf_stat_map(
            hemi_mesh,
            stat_map=hemi_data,
            bg_map=hemi_bg,
            hemi=hemi_label,
            view="lateral",
            colorbar=False,
            threshold=0.01,
            cmap="cold_hot",
            axes=ax,
        )

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.02,
                facecolor="#f8f9fc", transparent=False)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def render_tribe_brain_b64(pooled: np.ndarray, **kwargs) -> str:
    """Convenience: return the PNG as a data-URI-ready base64 string."""
    png = render_tribe_brain_png(pooled, **kwargs)
    return base64.b64encode(png).decode("ascii")
