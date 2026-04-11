"""
Decoders for fMRI data from TRIBE v2 encodings.

Decoder 1: Object classification (8-way softmax)
    Classes: Building, Skyscraper, Bridge, Street, Tree, River, Light, Factory

Decoder 2: Binary size classification (large=1, small=0)

Architecture:
    - PCA dimensionality reduction (~20,000 -> N_COMPONENTS)
    - Logistic regression with appropriate multi_class setting
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

N_COMPONENTS = 40

OBJECT_CLASSES = [
    "Building",
    "Skyscraper",
    "Bridge",
    "Street",
    "Tree",
    "River",
    "Light",
    "Factory",
]

SIZE_CLASSES = {0: "small", 1: "large"}


def build_object_decoder() -> Pipeline:
    """
    8-way object classification decoder.

    Pipeline: StandardScaler -> PCA(N_COMPONENTS) -> LogisticRegression (softmax)
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=N_COMPONENTS, random_state=42)),
            (
                "clf",
                LogisticRegression(
                    multi_class="multinomial",
                    solver="lbfgs",
                    max_iter=1000,
                    C=1.0,
                    random_state=42,
                ),
            ),
        ]
    )


def build_size_decoder() -> Pipeline:
    """
    Binary size classification decoder (0=small, 1=large).

    Pipeline: StandardScaler -> PCA(N_COMPONENTS) -> LogisticRegression (binary)
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=N_COMPONENTS, random_state=42)),
            (
                "clf",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=1000,
                    C=1.0,
                    random_state=42,
                ),
            ),
        ]
    )
