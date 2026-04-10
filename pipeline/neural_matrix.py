"""CSV sentences → TRIBE neural preds → matrix ``X`` + size/element labels. See ``--help``."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from tribe.model import load_model, predict_from_text_string

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ROW_CACHE_VERSION = 1


def parse_city_label(combined: str) -> tuple[str, str]:
    """Split ``\"small building\"`` into (\"small\", \"building\")."""
    s = (combined or "").strip().lower()
    if not s:
        raise ValueError("empty label")
    parts = s.split(None, 1)
    if len(parts) < 2:
        raise ValueError(f"expected '<size> <element>', got: {combined!r}")
    size, element = parts[0], parts[1]
    return size, element


def labels_to_indices(
    values: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Map strings to contiguous int ids; return (indices, unique_names_sorted)."""
    uniques = sorted(set(values))
    name_to_id = {n: i for i, n in enumerate(uniques)}
    idx = np.array([name_to_id[v] for v in values], dtype=np.int64)
    return idx, uniques


def _row_shard_path(row_cache_dir: Path, index: int) -> Path:
    return row_cache_dir / f"{index:05d}.npz"


def _load_row_shard(
    path: Path,
    *,
    expected_text: str,
    expected_n_vertices: int | None = None,
) -> tuple[np.ndarray, int] | None:
    """Return (pooled, n_segments) if shard matches *expected_text*, else None."""
    try:
        z = np.load(path, allow_pickle=True)
    except Exception as e:
        logger.warning("Ignoring bad cache shard %s: %s", path, e)
        return None

    ver = int(z["cache_version"]) if "cache_version" in z.files else 0
    if ver != ROW_CACHE_VERSION:
        return None

    cached_text = str(z["text"].item()) if z["text"].shape else str(z["text"])
    if cached_text.strip() != (expected_text or "").strip():
        return None

    nv = int(z["n_vertices"].item()) if "n_vertices" in z.files else None
    if expected_n_vertices is not None and nv is not None and nv != expected_n_vertices:
        logger.warning(
            "Ignoring cache shard %s (n_vertices %s != %s)",
            path,
            nv,
            expected_n_vertices,
        )
        return None

    pooled = np.asarray(z["pooled"], dtype=np.float32)
    n_seg = int(z["n_segments"].item())
    return pooled, n_seg


def _save_row_shard(
    path: Path,
    *,
    pooled: np.ndarray,
    n_segments: int,
    text: str,
    n_vertices: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        suffix=".npz", dir=path.parent, text=False
    )
    try:
        os.close(fd)
        tmp_path = Path(tmp_name)
        np.savez_compressed(
            tmp_path,
            cache_version=np.int32(ROW_CACHE_VERSION),
            pooled=np.asarray(pooled, dtype=np.float32),
            n_segments=np.int64(n_segments),
            n_vertices=np.int64(n_vertices),
            text=np.array(text, dtype=object),
        )
        tmp_path.replace(path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def build_neural_matrix(
    csv_path: Path,
    *,
    cache_folder: str | None = None,
    limit: int | None = None,
    verbose_tribe: bool = False,
    row_cache_dir: Path | None = None,
    force_recompute: bool = False,
) -> dict:
    """Load CSV, run TRIBE per row, return arrays and metadata dict.

    If *row_cache_dir* is set, each successful row is written immediately to
    ``{row_cache_dir}/{index:05d}.npz``. Shards are keyed by row index and
    validated by *text* so reruns skip completed sentences. Labels in the CSV
    can change without invalidating cached neural vectors.
    """
    df = pd.read_csv(csv_path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must have columns: text, label")

    rows = df.dropna(subset=["text", "label"])
    if limit is not None:
        rows = rows.head(limit)

    texts = rows["text"].astype(str).tolist()
    raw_labels = rows["label"].astype(str).tolist()

    sizes: list[str] = []
    elements: list[str] = []
    for lab in raw_labels:
        sz, el = parse_city_label(lab)
        sizes.append(sz)
        elements.append(el)

    use_cache = row_cache_dir is not None
    if use_cache:
        row_cache_dir = row_cache_dir.resolve()
        row_cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Row cache directory: %s", row_cache_dir)

    model = None
    expected_nv: int | None = None
    pooled_list: list[np.ndarray] = []
    n_segments_list: list[int] = []

    for i, text in enumerate(texts):
        shard_path = _row_shard_path(row_cache_dir, i) if use_cache else None

        hit: tuple[np.ndarray, int] | None = None
        if use_cache and shard_path is not None and not force_recompute:
            hit = _load_row_shard(
                shard_path,
                expected_text=text,
                expected_n_vertices=expected_nv,
            )

        if hit is not None:
            pooled, n_seg = hit
            if expected_nv is None:
                expected_nv = int(pooled.shape[0])
            elif pooled.shape[0] != expected_nv:
                logger.warning(
                    "Ignoring cache shard %s (pooled dim %s != %s)",
                    shard_path.name if shard_path else "?",
                    pooled.shape[0],
                    expected_nv,
                )
                hit = None

        if hit is not None:
            pooled, n_seg = hit
            logger.info(
                "Cache hit %d/%d (segments=%d, dim=%d)",
                i + 1,
                len(texts),
                n_seg,
                pooled.shape[0],
            )
        else:
            if model is None:
                logger.info(
                    "Loading TRIBE (first run may download ~2GB weights)..."
                )
                model = load_model(cache_folder=cache_folder)

            pooled, preds, _ = predict_from_text_string(
                model, text, verbose=verbose_tribe
            )
            n_seg = int(preds.shape[0])
            if expected_nv is None:
                expected_nv = int(pooled.shape[0])
            elif pooled.shape[0] != expected_nv:
                raise RuntimeError(
                    f"row {i}: n_vertices {pooled.shape[0]} != {expected_nv} "
                    "(clear row cache if you switched TRIBE checkpoints)"
                )

            if use_cache and shard_path is not None:
                _save_row_shard(
                    shard_path,
                    pooled=pooled,
                    n_segments=n_seg,
                    text=text,
                    n_vertices=expected_nv,
                )
                logger.info(
                    "Cached row %d/%d -> %s",
                    i + 1,
                    len(texts),
                    shard_path.name,
                )

            logger.info(
                "Processed %d/%d (segments=%d, dim=%d)",
                i + 1,
                len(texts),
                n_seg,
                pooled.shape[0],
            )

        pooled_list.append(pooled)
        n_segments_list.append(n_seg)

    X = np.stack(pooled_list, axis=0).astype(np.float32, copy=False)
    y_size, size_classes = labels_to_indices(sizes)
    y_element, element_classes = labels_to_indices(elements)

    return {
        "X": X,
        "y_size": y_size,
        "y_element": y_element,
        "size_classes": size_classes,
        "element_classes": element_classes,
        "texts": np.array(texts, dtype=object),
        "labels_combined": np.array(raw_labels, dtype=object),
        "n_segments_per_sentence": np.array(n_segments_list, dtype=np.int64),
        "n_vertices": int(X.shape[1]),
    }


def save_npz_bundle(path: Path, bundle: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "size_classes": bundle["size_classes"],
        "element_classes": bundle["element_classes"],
        "n_vertices": bundle["n_vertices"],
    }
    np.savez_compressed(
        path,
        X=bundle["X"],
        y_size=bundle["y_size"],
        y_element=bundle["y_element"],
        texts=bundle["texts"],
        labels_combined=bundle["labels_combined"],
        n_segments_per_sentence=bundle["n_segments_per_sentence"],
        meta_json=json.dumps(meta),
    )
    logger.info("Wrote %s (X shape %s)", path, bundle["X"].shape)


def load_npz_bundle(path: Path) -> dict:
    """Load bundle written by :func:`save_npz_bundle` (for training scripts)."""
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta_json"]))
    return {
        "X": z["X"],
        "y_size": z["y_size"],
        "y_element": z["y_element"],
        "texts": z["texts"],
        "labels_combined": z["labels_combined"],
        "n_segments_per_sentence": z["n_segments_per_sentence"],
        "size_classes": meta["size_classes"],
        "element_classes": meta["element_classes"],
        "n_vertices": meta["n_vertices"],
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--csv",
        type=Path,
        default=PROJECT_ROOT / "city_elements_dataset.csv",
        help="Input CSV with columns text, label",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "city_elements_neural.npz",
        help="Output .npz with X, y_size, y_element, texts, ...",
    )
    p.add_argument(
        "--cache-folder",
        default=None,
        help="TRIBE feature cache (default: ./cache under repo root)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N rows (for debugging)",
    )
    p.add_argument(
        "--verbose-tribe",
        action="store_true",
        help="Show TRIBE tqdm progress bars per sentence",
    )
    p.add_argument(
        "--row-cache-dir",
        type=Path,
        default=None,
        help="Per-row TRIBE outputs (default: <output_stem>_row_cache next to --output)",
    )
    p.add_argument(
        "--no-row-cache",
        action="store_true",
        help="Disable per-row resume cache",
    )
    p.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore existing row cache shards (still overwrites them as rows succeed)",
    )
    args = p.parse_args(argv)

    if not args.csv.is_file():
        logger.error("CSV not found: %s", args.csv)
        return 1

    row_cache_dir: Path | None = None
    if not args.no_row_cache:
        row_cache_dir = args.row_cache_dir
        if row_cache_dir is None:
            row_cache_dir = args.output.parent / f"{args.output.stem}_row_cache"

    bundle = build_neural_matrix(
        args.csv,
        cache_folder=args.cache_folder,
        limit=args.limit,
        verbose_tribe=args.verbose_tribe,
        row_cache_dir=row_cache_dir,
        force_recompute=args.force_recompute,
    )
    save_npz_bundle(args.output, bundle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
