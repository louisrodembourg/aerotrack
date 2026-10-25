"""VisDrone2019-DET -> YOLO labels (training) + neutral ground truth JSON (evaluation).

Raw annotation row (VisDrone-Dataset README):
    bbox_left, bbox_top, bbox_width, bbox_height, score, object_category, truncation, occlusion
In ground truth, `score` is 1 when the box is evaluated and 0 when it is ignored.

Output layout (data/processed/visdrone_det/):
    images/<split>/*.jpg     relative symlinks to raw images (raw stays untouched)
    labels/<split>/*.txt     YOLO `cls cx cy w h`, normalised, classes 1..10 -> 0..9
    ignore/<split>/*.txt     ignored regions `x y w h` in pixels, read by the training
                             dataset to mask them (aerotrack.train)
    gt/<split>.json          SplitGT in the VisDrone vocabulary
    stats/<split>.json       counts, including every dropped or altered box
    dataset.yaml             Ultralytics dataset file
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from aerotrack.data.classes import get_class_map
from aerotrack.data.gt import ImageGT, SplitGT
from aerotrack.paths import PROCESSED_DIR, RAW_DIR, data_relative

DATASET = "visdrone_det"


def parse_annotation(path: Path, ncols: int = 8) -> np.ndarray:
    """Rows as an (N, ncols) int array. Some VisDrone files end lines with a trailing comma,
    hence the filtering of empty fields."""
    rows = []
    for line in Path(path).read_text().splitlines():
        fields = [f for f in line.strip().split(",") if f.strip() != ""]
        if not fields:
            continue
        if len(fields) < ncols:
            raise ValueError(f"{path}: expected {ncols} fields, got {len(fields)}: {line!r}")
        rows.append([int(float(f)) for f in fields[:ncols]])
    return np.asarray(rows, dtype=np.int64).reshape(-1, ncols)


def clip_boxes(xywh: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Clip to the image. Returns clipped boxes and a validity mask (positive area)."""
    b = np.asarray(xywh, dtype=float).reshape(-1, 4)
    x0 = np.clip(b[:, 0], 0, width)
    y0 = np.clip(b[:, 1], 0, height)
    x1 = np.clip(b[:, 0] + b[:, 2], 0, width)
    y1 = np.clip(b[:, 1] + b[:, 3], 0, height)
    out = np.stack([x0, y0, x1 - x0, y1 - y0], axis=1)
    return out, (out[:, 2] > 0) & (out[:, 3] > 0)


def yolo_lines(xywh: np.ndarray, classes: np.ndarray, width: int, height: int) -> str:
    lines = []
    for (x, y, w, h), c in zip(xywh, classes, strict=True):
        cx, cy = (x + w / 2) / width, (y + h / 2) / height
        lines.append(f"{int(c)} {cx:.6f} {cy:.6f} {w / width:.6f} {h / height:.6f}\n")
    return "".join(lines)


def _relative_symlink(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(os.path.relpath(target, link.parent))


def convert_split(raw_split_dir: Path, out_dir: Path, split: str) -> dict:
    cmap = get_class_map("visdrone")
    ignore_ids = set(cmap.ignore)
    images_dir, ann_dir = raw_split_dir / "images", raw_split_dir / "annotations"
    image_files = sorted(images_dir.glob("*.jpg"))
    if not image_files:
        raise FileNotFoundError(f"no images in {images_dir}")

    stats: Counter = Counter()
    per_class: Counter = Counter()
    split_gt = SplitGT(DATASET, split, "visdrone", [])
    for image_id, img_path in enumerate(image_files, start=1):
        with Image.open(img_path) as im:
            width, height = im.size
        rows = parse_annotation(ann_dir / f"{img_path.stem}.txt")
        xywh, score, cat = rows[:, 0:4], rows[:, 4], rows[:, 5]

        is_region = np.isin(cat, list(ignore_ids))
        # A real class with score 0 is, per the README, "ignored in evaluation": treated
        # exactly like an ignored region rather than silently trained on or dropped.
        is_flagged = ~is_region & (score == 0)
        is_object = ~is_region & ~is_flagged
        unknown = set(cat[is_object].tolist()) - set(cmap.mapping)
        if unknown:
            raise ValueError(f"{img_path.name}: unknown VisDrone categories {unknown}")

        clipped, valid = clip_boxes(xywh[is_object], width, height)
        changed = np.any(np.abs(clipped - xywh[is_object]) > 0, axis=1)
        boxes, classes = clipped[valid], cat[is_object][valid]
        ignore = np.concatenate([xywh[is_region], xywh[is_flagged]]).astype(float)

        stats["images"] += 1
        stats["rows"] += len(rows)
        stats["objects_kept"] += int(valid.sum())
        stats["objects_clipped"] += int((changed & valid).sum())
        stats["objects_degenerate_dropped"] += int((~valid).sum())
        stats["ignored_regions_cat0"] += int((cat == 0).sum())
        stats["others_cat11"] += int((cat == 11).sum())
        stats["score0_objects_to_ignore"] += int(is_flagged.sum())
        stats["images_with_ignore"] += int(len(ignore) > 0)
        per_class.update(cmap.mapping[int(c)] for c in classes)

        name = img_path.name
        link = out_dir / "images" / split / name
        _relative_symlink(img_path, link)
        label = out_dir / "labels" / split / f"{img_path.stem}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text(yolo_lines(boxes, classes - 1, width, height))
        ign = out_dir / "ignore" / split / f"{img_path.stem}.txt"
        ign.parent.mkdir(parents=True, exist_ok=True)
        ign.write_text("".join(f"{x:g} {y:g} {w:g} {h:g}\n" for x, y, w, h in ignore))

        split_gt.images.append(ImageGT(
            id=image_id, file=data_relative(img_path), width=width, height=height,
            boxes=boxes, classes=classes, ignore=ignore,
        ))

    split_gt.save(out_dir / "gt" / f"{split}.json")
    result = {**stats, "per_class": dict(sorted(per_class.items()))}
    (out_dir / "stats").mkdir(parents=True, exist_ok=True)
    (out_dir / "stats" / f"{split}.json").write_text(json.dumps(result, indent=2))
    return result


def write_dataset_yaml(out_dir: Path, splits: list[str]) -> Path:
    names = get_class_map("visdrone").mapping
    cfg = {
        "path": str(out_dir.resolve()),
        **{("test" if s == "test-dev" else s): f"images/{s}" for s in splits},
        "names": {sid - 1: name for sid, name in sorted(names.items())},
    }
    path = out_dir / "dataset.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def convert(cfg: dict, raw_root: Path | None = None, out_dir: Path | None = None) -> dict:
    raw_root = raw_root or RAW_DIR / DATASET
    out_dir = out_dir or PROCESSED_DIR / DATASET
    report = {}
    for split, spec in cfg["splits"].items():
        stats = convert_split(raw_root / spec["dir"], out_dir, split)
        expected = spec.get("expected_images")
        if expected is not None and stats["images"] != expected:
            raise RuntimeError(
                f"{split}: {stats['images']} images, the VisDrone README announces {expected}. "
                "The archive is incomplete or not the official release."
            )
        report[split] = stats
        print(f"visdrone_det/{split}: {stats['images']} images, {stats['objects_kept']} objects, "
              f"{stats['ignored_regions_cat0'] + stats['others_cat11']} ignore regions")
    write_dataset_yaml(out_dir, list(cfg["splits"]))
    return report
