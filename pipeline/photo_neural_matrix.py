"""
Still images → short looped MP4 → TRIBE → neural matrix.

Layout (under ``--dataset-root``, default ``data/photo_dataset``)::

    source/bridge/
    source/lake/
    source/skyscrapers/
    source/trees/
    generated_videos/   # auto-written MP4s (gitignored)

Writes ``outputs/photo_tribe_neural.npz`` (same bundle keys as ``neural_matrix`` for
``train_element_classifier``) and optional row-cache shards for resume.

By default, if ``--output`` already exists, **merges** with it: rows for images still under
``source/`` are reused (no second TRIBE pass), **new** images are processed and appended,
and rows for files you removed from disk are **kept**. Pass ``--no-merge-output`` to build
only from the current tree (replace the matrix).

Use ``--holdout-per-class K`` (with ``--holdout-seed``) to randomly reserve **K** images per
class for ``--holdout-output`` (evaluation only). The main ``--output`` matrix contains the
rest — use it for ``train_element_classifier``; **do not** train on the holdout file.

By default skips **Whisper/ASR** on video audio (``TRIBE_VIDEO_SKIP_WHISPER=1``) for speed;
pass ``--video-whisper`` to run the full tribev2 word pipeline.

By default loads **video modality only** for TRIBE (``TRIBE_FEATURES_VIDEO_ONLY=1``) so text/audio
encoders are not prepared; pass ``--tribe-all-modalities`` for full multimodal extractors.

Requires ``ffmpeg`` on PATH.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from pipeline.neural_matrix import labels_to_indices, normalize_class_label
from tribe.model import load_model, predict_from_video_pooled

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PHOTO_ROW_CACHE_VERSION = 1

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

DEFAULT_CLASSES = ("bridge", "lake", "skyscrapers", "trees")


def _sample_key(label: str, rel_posix: str) -> str:
    return f"{label}|{rel_posix}"


def _shard_name(source_key: str) -> str:
    digest = hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:24]
    return f"{digest}.npz"


def _load_photo_shard(
    path: Path,
    *,
    expected_key: str,
    expected_n_vertices: int | None,
) -> tuple[np.ndarray, int] | None:
    try:
        z = np.load(path, allow_pickle=True)
    except Exception as e:
        logger.warning("Ignoring bad photo shard %s: %s", path, e)
        return None
    ver = int(z["cache_version"]) if "cache_version" in z.files else 0
    if ver != PHOTO_ROW_CACHE_VERSION:
        return None
    cached = str(z["source_key"].item()) if z["source_key"].shape else str(z["source_key"])
    if cached.strip() != (expected_key or "").strip():
        return None
    nv = int(z["n_vertices"].item()) if "n_vertices" in z.files else None
    if expected_n_vertices is not None and nv is not None and nv != expected_n_vertices:
        logger.warning(
            "Ignoring photo shard %s (n_vertices %s != %s)",
            path.name,
            nv,
            expected_n_vertices,
        )
        return None
    pooled = np.asarray(z["pooled"], dtype=np.float32)
    n_seg = int(z["n_segments"].item())
    return pooled, n_seg


def _save_photo_shard(
    path: Path,
    *,
    pooled: np.ndarray,
    n_segments: int,
    source_key: str,
    source_image: str,
    generated_video: str,
    n_vertices: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(suffix=".npz", dir=path.parent, text=False)
    try:
        os.close(fd)
        tmp_path = Path(tmp_name)
        np.savez_compressed(
            tmp_path,
            cache_version=np.int32(PHOTO_ROW_CACHE_VERSION),
            pooled=np.asarray(pooled, dtype=np.float32),
            n_segments=np.int64(n_segments),
            n_vertices=np.int64(n_vertices),
            source_key=np.array(source_key, dtype=object),
            source_image=np.array(source_image, dtype=object),
            generated_video=np.array(generated_video, dtype=object),
        )
        tmp_path.replace(path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _check_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it (e.g. brew install ffmpeg) "
            "to convert images to MP4."
        )
    return exe


def image_to_looped_mp4(
    *,
    image_path: Path,
    out_mp4: Path,
    duration_sec: float,
    fps: int,
    ffmpeg_exe: str,
) -> None:
    """Encode a still image as an H.264 MP4 with silent stereo AAC (TRIBE extracts audio)."""
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    vf = f"fps={fps},format=yuv420p,scale=trunc(iw/2)*2:trunc(ih/2)*2"
    cmd = [
        ffmpeg_exe,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-loop",
        "1",
        "-i",
        str(image_path),
        "-f",
        "lavfi",
        "-i",
        "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-t",
        str(duration_sec),
        "-c:v",
        "libx264",
        "-vf",
        vf,
        "-c:a",
        "aac",
        "-shortest",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True)


def _load_photo_npz_rows(path: Path) -> tuple[dict[str, dict], int] | None:
    """Load existing photo matrix .npz into ``source_key -> row`` and ``n_vertices``."""
    try:
        z = np.load(path, allow_pickle=True)
    except Exception as e:
        logger.warning("Could not read existing output for merge: %s", e)
        return None
    need = ("X", "texts", "labels_combined", "n_segments_per_sentence")
    if not all(k in z.files for k in need):
        return None
    X = np.asarray(z["X"], dtype=np.float32)
    if X.ndim != 2 or X.shape[0] == 0:
        return None
    n_vertices = int(X.shape[1])
    texts = z["texts"]
    labels = z["labels_combined"]
    n_seg = np.asarray(z["n_segments_per_sentence"], dtype=np.int64)
    n = X.shape[0]
    if len(texts) != n or len(labels) != n or n_seg.shape[0] != n:
        logger.warning("Existing output has mismatched row lengths; not merging.")
        return None
    out: dict[str, dict] = {}
    for i in range(n):
        label = str(labels[i])
        rel = str(texts[i])
        norm = normalize_class_label(label)
        sk = _sample_key(norm, rel)
        out[sk] = {
            "pooled": np.asarray(X[i], dtype=np.float32, copy=True),
            "n_segments": int(n_seg[i]),
            "label": norm,
            "rel_posix": rel,
        }
    return out, n_vertices


def iter_photo_samples(
    *,
    dataset_root: Path,
    source_subdir: str,
    class_names: tuple[str, ...] | None = None,
) -> list[tuple[str, Path, str, str]]:
    """Return sorted list of (label, image_path, rel_posix, source_key)."""
    source_root = (dataset_root / source_subdir).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source root not found: {source_root}")

    names = class_names if class_names is not None else DEFAULT_CLASSES
    out: list[tuple[str, Path, str, str]] = []
    for label in names:
        folder = source_root / label
        if not folder.is_dir():
            logger.warning("Missing class folder (skip): %s", folder)
            continue
        norm = normalize_class_label(label)
        for p in sorted(folder.rglob("*")):
            if not p.is_file():
                continue
            if p.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                rel = p.resolve().relative_to(source_root)
            except ValueError:
                rel = Path(p.name)
            rel_posix = rel.as_posix()
            key = _sample_key(norm, rel_posix)
            out.append((norm, p.resolve(), rel_posix, key))
    out.sort(key=lambda t: (t[0], t[2]))
    seen: set[str] = set()
    deduped: list[tuple[str, Path, str, str]] = []
    for t in out:
        if t[3] in seen:
            logger.warning("Duplicate sample key (skip second path): %s", t[3])
            continue
        seen.add(t[3])
        deduped.append(t)
    return deduped


def split_holdout_per_class(
    samples: list[tuple[str, Path, str, str]],
    holdout_per_class: int,
    seed: int,
) -> tuple[list[tuple[str, Path, str, str]], list[tuple[str, Path, str, str]]]:
    """Random but reproducible split: ``holdout_per_class`` samples per label → hold list."""
    if holdout_per_class <= 0:
        return samples, []

    from collections import defaultdict

    by_label: dict[str, list[tuple[str, Path, str, str]]] = defaultdict(list)
    for t in samples:
        by_label[t[0]].append(t)

    train: list[tuple[str, Path, str, str]] = []
    hold: list[tuple[str, Path, str, str]] = []
    for label in sorted(by_label.keys()):
        lst = sorted(by_label[label], key=lambda x: x[2])
        if len(lst) <= holdout_per_class:
            raise RuntimeError(
                f"class {label!r}: need more than {holdout_per_class} images (have {len(lst)}) "
                "to reserve holdout and keep at least one training image per class"
            )
        h = int(hashlib.sha256(f"{seed}\0{label}".encode()).hexdigest()[:15], 16)
        rng = random.Random(h)
        shuffled = lst.copy()
        rng.shuffle(shuffled)
        hold.extend(shuffled[:holdout_per_class])
        train.extend(shuffled[holdout_per_class:])

    train.sort(key=lambda t: (t[0], t[2]))
    hold.sort(key=lambda t: (t[0], t[2]))
    return train, hold


def build_photo_neural_bundle(
    *,
    dataset_root: Path,
    source_subdir: str,
    generated_subdir: str,
    cache_folder: str | None,
    row_cache_dir: Path | None,
    force_recompute: bool,
    force_reencode: bool,
    duration_sec: float,
    fps: int,
    verbose_tribe: bool,
    limit: int | None,
    class_names: tuple[str, ...] | None,
    merge_output_path: Path | None,
    merge_output: bool,
    samples: list[tuple[str, Path, str, str]] | None = None,
    drop_merge_keys: set[str] | None = None,
) -> dict:
    ffmpeg_exe = _check_ffmpeg()
    if samples is None:
        samples = iter_photo_samples(
            dataset_root=dataset_root,
            source_subdir=source_subdir,
            class_names=class_names,
        )
        if limit is not None:
            samples = samples[:limit]
    else:
        samples = list(samples)
    if not samples and not (
        merge_output and merge_output_path is not None and merge_output_path.is_file()
    ):
        raise RuntimeError(
            f"No images found under {dataset_root / source_subdir}. "
            f"Add files ({', '.join(sorted(IMAGE_EXTENSIONS))}) to class subfolders."
        )

    merged_rows: dict[str, dict] = {}
    if merge_output and merge_output_path is not None and merge_output_path.is_file():
        loaded = _load_photo_npz_rows(merge_output_path)
        if loaded is not None:
            merged_rows, nv0 = loaded
            logger.info(
                "Merge: loaded %d rows from %s (n_vertices=%d)",
                len(merged_rows),
                merge_output_path,
                nv0,
            )
    if drop_merge_keys:
        for k in drop_merge_keys:
            merged_rows.pop(k, None)

    use_cache = row_cache_dir is not None
    if use_cache:
        row_cache_dir = row_cache_dir.resolve()
        row_cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Photo row cache: %s", row_cache_dir)

    model = None
    expected_nv: int | None = None
    if merged_rows:
        any_row = next(iter(merged_rows.values()))
        expected_nv = int(any_row["pooled"].shape[0])
    pooled_list: list[np.ndarray] = []
    n_seg_list: list[int] = []
    texts: list[str] = []
    raw_labels: list[str] = []
    gen_root = (dataset_root / generated_subdir).resolve()

    for idx, (label, img_path, rel_posix, source_key) in enumerate(samples):
        out_mp4 = gen_root / Path(rel_posix).with_suffix(".mp4")
        shard_path = (row_cache_dir / _shard_name(source_key)) if use_cache else None
        hit: tuple[np.ndarray, int] | None = None
        from_prior_output = False

        if merge_output and source_key in merged_rows and not force_recompute:
            row = merged_rows[source_key]
            pooled = np.asarray(row["pooled"], dtype=np.float32)
            n_seg = int(row["n_segments"])
            if expected_nv is None:
                expected_nv = int(pooled.shape[0])
            elif pooled.shape[0] != expected_nv:
                raise RuntimeError(
                    f"{rel_posix}: merged output n_vertices {pooled.shape[0]} != {expected_nv}"
                )
            hit = (pooled, n_seg)
            from_prior_output = True
            logger.info(
                "Existing output row %d/%d %s (segments=%d, skip encode & TRIBE)",
                idx + 1,
                len(samples),
                rel_posix,
                n_seg,
            )

        if hit is None:
            if force_reencode and out_mp4.is_file():
                out_mp4.unlink()
            if not out_mp4.is_file():
                try:
                    log_rel = out_mp4.relative_to(dataset_root)
                except ValueError:
                    log_rel = out_mp4
                logger.info("Encoding %d/%d → %s", idx + 1, len(samples), log_rel)
                image_to_looped_mp4(
                    image_path=img_path,
                    out_mp4=out_mp4,
                    duration_sec=duration_sec,
                    fps=fps,
                    ffmpeg_exe=ffmpeg_exe,
                )

        if hit is None and use_cache and shard_path is not None and not force_recompute:
            hit = _load_photo_shard(
                shard_path,
                expected_key=source_key,
                expected_n_vertices=expected_nv,
            )
            if hit is not None:
                pooled, n_seg = hit
                if expected_nv is None:
                    expected_nv = int(pooled.shape[0])
                elif pooled.shape[0] != expected_nv:
                    logger.warning("Shard dim mismatch; recomputing %s", shard_path.name)
                    hit = None

        if hit is not None and not from_prior_output:
            pooled, n_seg = hit
            logger.info(
                "Row-cache hit %d/%d %s (segments=%d)",
                idx + 1,
                len(samples),
                rel_posix,
                n_seg,
            )
        elif hit is None:
            if model is None:
                logger.info("Loading TRIBE (first run may download weights)...")
                model = load_model(cache_folder=cache_folder)
            pooled, preds, _ = predict_from_video_pooled(
                model,
                str(out_mp4),
                verbose=verbose_tribe,
            )
            n_seg = int(preds.shape[0])
            if expected_nv is None:
                expected_nv = int(pooled.shape[0])
            elif pooled.shape[0] != expected_nv:
                raise RuntimeError(
                    f"sample {rel_posix}: n_vertices {pooled.shape[0]} != {expected_nv}"
                )
            if use_cache and shard_path is not None:
                _save_photo_shard(
                    shard_path,
                    pooled=pooled,
                    n_segments=n_seg,
                    source_key=source_key,
                    source_image=str(img_path),
                    generated_video=str(out_mp4),
                    n_vertices=expected_nv,
                )
                logger.info("Cached photo shard %s", shard_path.name)
            logger.info(
                "Processed %d/%d %s (segments=%d, dim=%d)",
                idx + 1,
                len(samples),
                rel_posix,
                n_seg,
                pooled.shape[0],
            )

        merged_rows[source_key] = {
            "pooled": np.asarray(pooled, dtype=np.float32, copy=True),
            "n_segments": int(n_seg),
            "label": label,
            "rel_posix": rel_posix,
        }

    if not merged_rows:
        raise RuntimeError("No rows to write (empty merge and no samples).")

    key_order = sorted(
        merged_rows.keys(),
        key=lambda k: (merged_rows[k]["label"], merged_rows[k]["rel_posix"]),
    )
    pooled_list = [merged_rows[k]["pooled"] for k in key_order]
    n_seg_list = [merged_rows[k]["n_segments"] for k in key_order]
    texts = [merged_rows[k]["rel_posix"] for k in key_order]
    raw_labels = [merged_rows[k]["label"] for k in key_order]

    sizes = ["na"] * len(raw_labels)
    X = np.stack(pooled_list, axis=0).astype(np.float32, copy=False)
    y_size, size_classes = labels_to_indices(sizes)
    y_element, element_classes = labels_to_indices(raw_labels)

    return {
        "X": X,
        "y_size": y_size,
        "y_element": y_element,
        "size_classes": size_classes,
        "element_classes": element_classes,
        "texts": np.array(texts, dtype=object),
        "labels_combined": np.array(raw_labels, dtype=object),
        "n_segments_per_sentence": np.array(n_seg_list, dtype=np.int64),
        "n_vertices": int(X.shape[1]),
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "photo_dataset",
        help="Root containing source/ and generated_videos/",
    )
    p.add_argument(
        "--source-subdir",
        default="source",
        help="Subfolder with per-class image directories",
    )
    p.add_argument(
        "--generated-subdir",
        default="generated_videos",
        help="Where looped MP4s are written (under dataset-root)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "photo_tribe_neural.npz",
        help="Output .npz for train_element_classifier",
    )
    p.add_argument(
        "--row-cache-dir",
        type=Path,
        default=None,
        help="Per-sample shards (default: <output_stem>_row_cache next to --output)",
    )
    p.add_argument(
        "--no-row-cache",
        action="store_true",
        help="Disable resume cache",
    )
    p.add_argument(
        "--no-merge-output",
        action="store_true",
        help="Do not load/merge existing --output; matrix includes only current source images",
    )
    p.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore existing photo row-cache shards",
    )
    p.add_argument(
        "--force-reencode",
        action="store_true",
        help="Re-encode MP4s from images even if file exists",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Looped video length in seconds (still image held on screen; default 5)",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Output video frame rate for the still-image loop",
    )
    p.add_argument(
        "--cache-folder",
        default=None,
        help="TRIBE Hugging Face / checkpoint cache (default: ./cache)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N images (after sorting)",
    )
    p.add_argument(
        "--verbose-tribe",
        action="store_true",
        help="TRIBE tqdm progress",
    )
    p.add_argument(
        "--video-whisper",
        action="store_true",
        help="Run Whisper/transcription on video audio (slow; default skips ASR)",
    )
    p.add_argument(
        "--tribe-all-modalities",
        action="store_true",
        help="Prepare text+audio+video TRIBE extractors (heavy VRAM; default video-only)",
    )
    p.add_argument(
        "--classes",
        default=",".join(DEFAULT_CLASSES),
        help="Comma-separated class subfolder names under source/",
    )
    p.add_argument(
        "--holdout-per-class",
        type=int,
        default=0,
        metavar="K",
        help="Randomly reserve K images per class for --holdout-output (not for training); "
        "requires >K images per class",
    )
    p.add_argument(
        "--holdout-output",
        type=Path,
        default=None,
        help="Holdout .npz path (default: <output_stem>_holdout.npz next to --output)",
    )
    p.add_argument(
        "--holdout-seed",
        type=int,
        default=42,
        help="RNG seed for holdout selection (per-class derived seed)",
    )
    args = p.parse_args(argv)

    if args.video_whisper:
        os.environ["TRIBE_VIDEO_SKIP_WHISPER"] = "0"
    else:
        os.environ.setdefault("TRIBE_VIDEO_SKIP_WHISPER", "1")

    if args.tribe_all_modalities:
        os.environ["TRIBE_FEATURES_VIDEO_ONLY"] = "0"
    else:
        os.environ.setdefault("TRIBE_FEATURES_VIDEO_ONLY", "1")

    row_cache_dir: Path | None = None
    if not args.no_row_cache:
        row_cache_dir = args.row_cache_dir
        if row_cache_dir is None:
            row_cache_dir = args.output.parent / f"{args.output.stem}_row_cache"

    class_tuple = tuple(
        c.strip().lower()
        for c in args.classes.split(",")
        if c.strip()
    )
    if not class_tuple:
        class_tuple = DEFAULT_CLASSES

    holdout_n = max(0, int(args.holdout_per_class))
    holdout_path: Path | None = None
    if holdout_n > 0:
        holdout_path = (
            args.holdout_output
            if args.holdout_output is not None
            else args.output.parent / f"{args.output.stem}_holdout{args.output.suffix}"
        )

    try:
        all_samples = iter_photo_samples(
            dataset_root=args.dataset_root.resolve(),
            source_subdir=args.source_subdir,
            class_names=class_tuple,
        )
        if args.limit is not None:
            all_samples = all_samples[: args.limit]

        if holdout_n > 0:
            train_samples, hold_samples = split_holdout_per_class(
                all_samples, holdout_n, args.holdout_seed
            )
            hold_keys = {t[3] for t in hold_samples}
            train_keys = {t[3] for t in train_samples}
            logger.info(
                "Holdout split: %d train rows, %d holdout rows (K=%d per class, seed=%d)",
                len(train_samples),
                len(hold_samples),
                holdout_n,
                args.holdout_seed,
            )
        else:
            train_samples = all_samples
            hold_samples = []
            hold_keys = set()
            train_keys = set()

        bundle = build_photo_neural_bundle(
            dataset_root=args.dataset_root.resolve(),
            source_subdir=args.source_subdir,
            generated_subdir=args.generated_subdir,
            cache_folder=args.cache_folder,
            row_cache_dir=row_cache_dir,
            force_recompute=args.force_recompute,
            force_reencode=args.force_reencode,
            duration_sec=args.duration,
            fps=args.fps,
            verbose_tribe=args.verbose_tribe,
            limit=None,
            class_names=class_tuple,
            merge_output_path=args.output.resolve(),
            merge_output=not args.no_merge_output,
            samples=train_samples,
            drop_merge_keys=hold_keys if hold_keys else None,
        )

        bundle_hold = None
        if holdout_n > 0 and holdout_path is not None:
            bundle_hold = build_photo_neural_bundle(
                dataset_root=args.dataset_root.resolve(),
                source_subdir=args.source_subdir,
                generated_subdir=args.generated_subdir,
                cache_folder=args.cache_folder,
                row_cache_dir=row_cache_dir,
                force_recompute=args.force_recompute,
                force_reencode=args.force_reencode,
                duration_sec=args.duration,
                fps=args.fps,
                verbose_tribe=args.verbose_tribe,
                limit=None,
                class_names=class_tuple,
                merge_output_path=holdout_path.resolve(),
                merge_output=not args.no_merge_output,
                samples=hold_samples,
                drop_merge_keys=train_keys if train_keys else None,
            )
    except (RuntimeError, FileNotFoundError, subprocess.CalledProcessError) as e:
        logger.error("%s", e)
        return 1

    def meta_for(split: str) -> dict:
        m = {
            "dataset_root": str(args.dataset_root.resolve()),
            "source_subdir": args.source_subdir,
            "generated_subdir": args.generated_subdir,
            "duration_sec": args.duration,
            "fps": args.fps,
            "modality": "photo_video",
            "TRIBE_VIDEO_SKIP_WHISPER": os.environ.get("TRIBE_VIDEO_SKIP_WHISPER", ""),
            "TRIBE_FEATURES_VIDEO_ONLY": os.environ.get("TRIBE_FEATURES_VIDEO_ONLY", ""),
            "merge_output": not args.no_merge_output,
            "split": split,
        }
        if holdout_n > 0:
            m["holdout_per_class"] = holdout_n
            m["holdout_seed"] = args.holdout_seed
            m["holdout_output"] = str(holdout_path.resolve()) if holdout_path else ""
        return m

    path = args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    train_meta = meta_for("train")
    np.savez_compressed(
        path,
        X=bundle["X"],
        y_size=bundle["y_size"],
        y_element=bundle["y_element"],
        texts=bundle["texts"],
        labels_combined=bundle["labels_combined"],
        n_segments_per_sentence=bundle["n_segments_per_sentence"],
        meta_json=json.dumps(
            {
                "size_classes": bundle["size_classes"],
                "element_classes": bundle["element_classes"],
                "n_vertices": bundle["n_vertices"],
                "photo_pipeline": train_meta,
            }
        ),
    )
    logger.info("Wrote %s (X shape %s)", path, bundle["X"].shape)

    if bundle_hold is not None and holdout_path is not None:
        ho = holdout_path.resolve()
        ho.parent.mkdir(parents=True, exist_ok=True)
        hold_meta = meta_for("holdout")
        np.savez_compressed(
            ho,
            X=bundle_hold["X"],
            y_size=bundle_hold["y_size"],
            y_element=bundle_hold["y_element"],
            texts=bundle_hold["texts"],
            labels_combined=bundle_hold["labels_combined"],
            n_segments_per_sentence=bundle_hold["n_segments_per_sentence"],
            meta_json=json.dumps(
                {
                    "size_classes": bundle_hold["size_classes"],
                    "element_classes": bundle_hold["element_classes"],
                    "n_vertices": bundle_hold["n_vertices"],
                    "photo_pipeline": hold_meta,
                }
            ),
        )
        logger.info("Wrote holdout %s (X shape %s)", ho, bundle_hold["X"].shape)

    return 0


if __name__ == "__main__":
    sys.exit(main())
