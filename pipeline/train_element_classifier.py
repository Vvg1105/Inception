"""
Train a multinomial logistic regression on TRIBE neural features to predict the CSV
``label`` class (e.g. park, street, skyscraper) from ``city_elements_neural.npz``.

High-dimensional features (~20k vertices) and ~80 samples → optional **PCA** after
scaling, then strong L2 logistic regression and optional cross-validation.

    python -m pipeline.train_element_classifier \\
        --data outputs/city_elements_neural.npz \\
        --model-out outputs/element_logreg.joblib \\
        --pca-components 40

While ``neural_matrix`` is still running, train on shards written so far::

    python -m pipeline.train_element_classifier \\
        --from-row-cache outputs/city_elements_neural_row_cache \\
        --csv city_elements_dataset.csv \\
        --model-out outputs/element_logreg_partial.joblib
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipeline.neural_matrix import build_bundle_from_row_cache, load_npz_bundle

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _pca_component_cap(
    *,
    pca_components: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int,
    random_state: int,
) -> int:
    """Cap PCA dims so each CV fold has enough rows (sklearn PCA constraint)."""
    n_train, n_features = X_train.shape
    max_by_data = min(max(1, pca_components), n_train, n_features)
    if cv_folds < 2:
        return max_by_data
    y_int = np.asarray(y_train, dtype=np.int64)
    counts = np.bincount(y_int)
    min_class = int(counts.min()) if counts.size else 0
    # StratifiedKFold needs n_splits <= each class count on y_train.
    n_splits_eff = min(cv_folds, min_class)
    if n_splits_eff < 2:
        logger.warning(
            "PCA cap: skipping stratified fold sizing (rarest class has %d train samples; "
            "need ≥2 per class for CV). Using data/feature cap only.",
            min_class,
        )
        return max_by_data
    if n_splits_eff < cv_folds:
        logger.info(
            "PCA cap: using %d stratified folds (requested %d; limited by smallest class=%d)",
            n_splits_eff,
            cv_folds,
            min_class,
        )
    skf = StratifiedKFold(
        n_splits=n_splits_eff,
        shuffle=True,
        random_state=random_state,
    )
    min_fold_train = min(len(tr) for tr, _ in skf.split(X_train, y_train))
    max_by_cv = min(min_fold_train, n_features)
    return min(max_by_data, max_by_cv)


def _make_pipeline(
    *,
    use_pca: bool,
    pca_n: int,
    C: float,
    random_state: int,
) -> Pipeline:
    steps: list = [("scaler", StandardScaler())]
    if use_pca:
        steps.append(
            (
                "pca",
                PCA(n_components=max(1, pca_n), random_state=random_state),
            )
        )
    steps.append(
        (
            "lr",
            LogisticRegression(
                C=C,
                max_iter=50_000,
                solver="lbfgs",
                random_state=random_state,
            ),
        )
    )
    return Pipeline(steps)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "city_elements_neural.npz",
        help="npz from pipeline.neural_matrix (ignored if --from-row-cache is set)",
    )
    p.add_argument(
        "--from-row-cache",
        type=Path,
        default=None,
        help="Directory of per-row *.npz shards; train on whatever is finished (parallel-safe)",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=PROJECT_ROOT / "city_elements_dataset.csv",
        help="Labels CSV (required with --from-row-cache)",
    )
    p.add_argument(
        "--model-out",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "element_logreg.joblib",
        help="Where to save fitted Pipeline (scaler + logistic)",
    )
    p.add_argument("--test-size", type=float, default=0.2, help="Holdout fraction")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument(
        "--C",
        type=float,
        default=0.01,
        help="Inverse L2 strength (smaller = more regularization; use ~1e-3–1 for p>>n)",
    )
    p.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Stratified CV folds on training split for mean accuracy (0 to skip)",
    )
    p.add_argument(
        "--pca-components",
        type=int,
        default=40,
        help="Number of PCA components after scaling (capped by train size and n_features)",
    )
    p.add_argument(
        "--no-pca",
        action="store_true",
        help="Disable PCA (only StandardScaler + logistic regression)",
    )
    args = p.parse_args(argv)

    data_source: str
    if args.from_row_cache is not None:
        if not args.csv.is_file():
            logger.error("CSV not found: %s", args.csv)
            return 1
        rc = args.from_row_cache
        if not rc.is_dir():
            logger.error("Row cache directory not found: %s", rc)
            return 1
        try:
            bundle = build_bundle_from_row_cache(args.csv, rc)
        except RuntimeError as e:
            logger.error("%s", e)
            return 1
        data_source = f"row_cache:{rc.resolve()}|csv:{args.csv.resolve()}"
        logger.info(
            "Loaded %d rows from row cache (partial dataset OK)",
            bundle["X"].shape[0],
        )
    else:
        if not args.data.is_file():
            logger.error("Data file not found: %s", args.data)
            return 1
        bundle = load_npz_bundle(args.data)
        data_source = str(args.data.resolve())

    X = np.asarray(bundle["X"], dtype=np.float64)
    y = np.asarray(bundle["y_element"], dtype=np.int64)
    names = list(bundle["element_classes"])

    logger.info("X shape %s, %d classes: %s", X.shape, len(names), names)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=y,
        )
    except ValueError as e:
        logger.warning("Stratified split unavailable (%s); using random split.", e)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=None,
        )

    use_pca = not args.no_pca
    pca_cap = _pca_component_cap(
        pca_components=args.pca_components,
        X_train=X_train,
        y_train=y_train,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
    )
    if use_pca and pca_cap < args.pca_components:
        logger.warning(
            "PCA n_components capped from %d to %d (CV fold train size / data)",
            args.pca_components,
            pca_cap,
        )
    clf = _make_pipeline(
        use_pca=use_pca,
        pca_n=pca_cap,
        C=args.C,
        random_state=args.random_state,
    )

    clf.fit(X_train, y_train)
    if use_pca and "pca" in clf.named_steps:
        pca = clf.named_steps["pca"]
        ev = float(np.sum(pca.explained_variance_ratio_))
        logger.info(
            "PCA: %d components, cumulative explained variance ratio ≈ %.4f",
            pca.n_components_,
            ev,
        )
    y_pred = clf.predict(X_test)
    logger.info(
        "Hold-out accuracy: %.3f",
        float((y_pred == y_test).mean()),
    )
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=names,
            labels=np.arange(len(names)),
            zero_division=0,
        )
    )

    if args.cv_folds >= 2:
        y_tr = np.asarray(y_train, dtype=np.int64)
        c = np.bincount(y_tr)
        min_cls = int(c.min()) if c.size else 0
        cv_splits = min(args.cv_folds, min_cls)
        if cv_splits < 2:
            logger.warning(
                "Skipping CV: smallest class has %d train samples (stratified k-fold needs ≥2)",
                min_cls,
            )
        else:
            if cv_splits < args.cv_folds:
                logger.info(
                    "CV: using %d folds (requested %d; capped by smallest class=%d)",
                    cv_splits,
                    args.cv_folds,
                    min_cls,
                )
            skf = StratifiedKFold(
                n_splits=cv_splits,
                shuffle=True,
                random_state=args.random_state,
            )
            cv_clf = _make_pipeline(
                use_pca=use_pca,
                pca_n=pca_cap,
                C=args.C,
                random_state=args.random_state,
            )
            scores = cross_val_score(
                cv_clf,
                X_train,
                y_train,
                cv=skf,
                scoring="accuracy",
            )
            logger.info(
                "CV accuracy on train (%d folds): %.3f ± %.3f",
                cv_splits,
                scores.mean(),
                scores.std(),
            )

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": clf,
            "element_classes": names,
            "n_vertices": bundle["n_vertices"],
            "data_path": data_source,
            "use_pca": use_pca,
            "pca_components_requested": args.pca_components,
            "pca_components_effective": pca_cap if use_pca else None,
        },
        args.model_out,
    )
    logger.info("Saved %s", args.model_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
