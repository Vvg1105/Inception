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

Requires ``ffmpeg`` on PATH.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
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
    return out


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
) -> dict:
    ffmpeg_exe = _check_ffmpeg()
    samples = iter_photo_samples(
        dataset_root=dataset_root,
        source_subdir=source_subdir,
        class_names=class_names,
    )
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise RuntimeError(
            f"No images found under {dataset_root / source_subdir}. "
            f"Add files ({', '.join(sorted(IMAGE_EXTENSIONS))}) to class subfolders."
        )

    use_cache = row_cache_dir is not None
    if use_cache:
        row_cache_dir = row_cache_dir.resolve()
        row_cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Photo row cache: %s", row_cache_dir)

    model = None
    expected_nv: int | None = None
    pooled_list: list[np.ndarray] = []
    n_seg_list: list[int] = []
    texts: list[str] = []
    raw_labels: list[str] = []
    gen_root = (dataset_root / generated_subdir).resolve()

    for idx, (label, img_path, rel_posix, source_key) in enumerate(samples):
        out_mp4 = gen_root / Path(rel_posix).with_suffix(".mp4")
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

        shard_path = (row_cache_dir / _shard_name(source_key)) if use_cache else None
        hit: tuple[np.ndarray, int] | None = None
        if use_cache and shard_path is not None and not force_recompute:
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

        if hit is not None:
            pooled, n_seg = hit
            logger.info(
                "Cache hit %d/%d %s (segments=%d)",
                idx + 1,
                len(samples),
                rel_posix,
                n_seg,
            )
        else:
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

        pooled_list.append(pooled)
        n_seg_list.append(n_seg)
        texts.append(rel_posix)
        raw_labels.append(label)

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
        "--classes",
        default=",".join(DEFAULT_CLASSES),
        help="Comma-separated class subfolder names under source/",
    )
    args = p.parse_args(argv)

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

    try:
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
            limit=args.limit,
            class_names=class_tuple,
        )
    except (RuntimeError, FileNotFoundError, subprocess.CalledProcessError) as e:
        logger.error("%s", e)
        return 1

    meta_extra = {
        "dataset_root": str(args.dataset_root.resolve()),
        "source_subdir": args.source_subdir,
        "generated_subdir": args.generated_subdir,
        "duration_sec": args.duration,
        "fps": args.fps,
        "modality": "photo_video",
    }
    bundle_meta = {
        "size_classes": bundle["size_classes"],
        "element_classes": bundle["element_classes"],
        "n_vertices": bundle["n_vertices"],
        "photo_pipeline": meta_extra,
    }
    path = args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=bundle["X"],
        y_size=bundle["y_size"],
        y_element=bundle["y_element"],
        texts=bundle["texts"],
        labels_combined=bundle["labels_combined"],
        n_segments_per_sentence=bundle["n_segments_per_sentence"],
        meta_json=json.dumps(bundle_meta),
    )
    logger.info("Wrote %s (X shape %s)", path, bundle["X"].shape)
    return 0


if __name__ == "__main__":
    sys.exit(main())
