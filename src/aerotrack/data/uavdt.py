"""UAVDT-Benchmark-M -> the same formats as VisDrone (YOLO + SplitGT for DET, TrackEval
benchmarks for MOT), plus per-sequence attributes for the week-11 robustness slices.

The toolkit ships no format description; everything below was read from its code and
checked on the data (see DECISIONS.md):

  GT/<seq>_gt_whole.txt   frame, id, x, y, w, h, out_of_view, occlusion, category
                          category 1 car, 2 truck, 3 bus. Used by the DET evaluation.
  GT/<seq>_gt.txt         frame, id, x, y, w, h, flag, 1, -1. Class-agnostic MOT ground
                          truth; rows with flag 0 are ignored (evaluateTracking.m). It is
                          NOT row-identical to gt_whole (20 of 50 sequences differ), so
                          classes are recovered by joining on (frame, id).
  GT/<seq>_gt_ignore.txt  frame, id, x, y, w, h, 1, -1, -1 ignored regions.
  M_attr/{train,test}/<seq>_attr.txt
                          daylight, night, fog; low-alt, medium-alt, high-alt;
                          front-view, side-view, bird-view; long-term (M_attr/readme.txt).
                          Views are not exclusive. One file name contains a stray space
                          ("M0701 _attr.txt").
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from aerotrack.data.classes import get_class_map
from aerotrack.data.gt import ImageGT, SplitGT
from aerotrack.data.ignore import UAVDT_DROP_COVERAGE
from aerotrack.data.mot import SequenceGT, write_benchmark
from aerotrack.data.visdrone_det import _relative_symlink, clip_boxes, yolo_lines
from aerotrack.paths import PROCESSED_DIR, RAW_DIR, data_relative

DATASET = "uavdt"
ATTRIBUTE_FLAGS = ["daylight", "night", "fog", "low-alt", "medium-alt", "high-alt",
                   "front-view", "side-view", "bird-view", "long-term"]


def read_rows(path: Path, ncols: int = 9) -> np.ndarray:
    text = path.read_text().strip() if path.exists() else ""
    if not text:
        return np.zeros((0, ncols))
    return np.loadtxt(text.splitlines(), delimiter=",", ndmin=2)[:, :ncols]


def read_attributes(attr_root: Path) -> dict[str, dict]:
    out = {}
    for split in ("train", "test"):
        for f in sorted((attr_root / split).glob("*.txt")):
            seq = f.name.replace(" ", "").removesuffix("_attr.txt")
            flags = [int(v) for v in f.read_text().strip().split(",")]
            if len(flags) != len(ATTRIBUTE_FLAGS):
                raise ValueError(f"{f}: {len(flags)} flags, expected {len(ATTRIBUTE_FLAGS)}")
            on = {name for name, v in zip(ATTRIBUTE_FLAGS, flags, strict=True) if v}
            out[seq] = {
                "split": split,
                "weather": [w for w in ("daylight", "night", "fog") if w in on],
                "altitude": [a.removesuffix("-alt") for a in ATTRIBUTE_FLAGS[3:6] if a in on],
                "view": [v.removesuffix("-view") for v in ATTRIBUTE_FLAGS[6:9] if v in on],
                "long_term": "long-term" in on,
            }
    return out


def _class_lookup(whole: np.ndarray) -> dict[tuple[int, int], int]:
    return {(int(f), int(t)): int(c) for f, t, c in whole[:, [0, 1, 8]]}


def convert_sequence(seq: str, split: str, raw: Path, out: Path, attrs: dict,
                     stats: Counter, split_gt: SplitGT) -> SequenceGT:
    cmap = get_class_map("uavdt")
    targets = cmap.targets
    image_dir = raw / "UAV-benchmark-M" / seq
    frames = sorted(image_dir.glob("img*.jpg"))
    with Image.open(frames[0]) as im:
        width, height = im.size
    gt_dir = raw / "UAV-benchmark-MOTD_v1.0" / "GT"
    whole = read_rows(gt_dir / f"{seq}_gt_whole.txt")
    mot = read_rows(gt_dir / f"{seq}_gt.txt")
    ign = read_rows(gt_dir / f"{seq}_gt_ignore.txt")
    if len(whole) and not set(np.unique(whole[:, 8]).astype(int)) <= set(cmap.mapping):
        raise ValueError(f"{seq}: unknown categories {np.unique(whole[:, 8])}")

    # --- DET: every frame is an image, labels from gt_whole --------------------------
    for fi, frame_path in enumerate(frames, start=1):
        rows = whole[whole[:, 0] == fi]
        boxes, valid = clip_boxes(rows[:, 2:6], width, height)
        classes = rows[valid, 8].astype(int)
        boxes = boxes[valid]
        frame_ign = ign[ign[:, 0] == fi, 2:6] if len(ign) else np.zeros((0, 4))
        stem = f"{seq}_{frame_path.stem}"
        _relative_symlink(frame_path, out / "det" / "images" / split / f"{stem}.jpg")
        label = out / "det" / "labels" / split / f"{stem}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text(yolo_lines(boxes, classes - 1, width, height))
        ign_file = out / "det" / "ignore" / split / f"{stem}.txt"
        ign_file.parent.mkdir(parents=True, exist_ok=True)
        ign_file.write_text("".join(f"{x:g} {y:g} {w:g} {h:g}\n" for x, y, w, h in frame_ign))
        split_gt.images.append(ImageGT(
            id=len(split_gt.images) + 1, file=data_relative(frame_path), width=width,
            height=height, boxes=boxes, classes=classes, ignore=frame_ign,
            extra={"sequence": seq, "frame": fi, **attrs.get(seq, {})},
        ))
        stats["det_images"] += 1
        stats["det_objects"] += len(boxes)
        stats["det_degenerate_dropped"] += int((~valid).sum())
        stats["det_clipped"] += int(np.any(np.abs(boxes - rows[valid, 2:6]) > 0, axis=1).sum())

    # --- MOT: official class-agnostic gt, classes joined from gt_whole ----------------
    flagged = mot[:, 6] == 0
    mot = mot[~flagged]
    lookup = _class_lookup(whole)
    cats = np.array([lookup.get((int(f), int(t)), -1) for f, t in mot[:, :2]], dtype=int)
    target_idx = np.array([targets.index(cmap.mapping[c]) if c in cmap.mapping else -1
                           for c in cats], dtype=float)
    stats["mot_rows"] += len(mot)
    stats["mot_rows_flag0_dropped"] += int(flagged.sum())
    stats["mot_rows_without_class"] += int((cats < 0).sum())
    stats["frames"] += len(frames)
    objects = np.column_stack([mot[:, :6], target_idx]) if len(mot) else np.zeros((0, 7))
    ignore = np.column_stack([ign[:, 0], ign[:, 2:6]]) if len(ign) else np.zeros((0, 5))
    return SequenceGT(seq, image_dir, len(frames), width, height, ".jpg", objects, ignore)


def convert(cfg: dict, raw_root: Path | None = None, out_dir: Path | None = None) -> dict:
    raw = raw_root or RAW_DIR / DATASET
    out = out_dir or PROCESSED_DIR / DATASET
    attrs = read_attributes(raw / "M_attr")
    (out).mkdir(parents=True, exist_ok=True)
    (out / "attributes.json").write_text(json.dumps(attrs, indent=2))
    targets = get_class_map("uavdt").targets
    report = {}
    for split, spec in cfg["splits"].items():
        seqs = sorted(s for s, a in attrs.items() if a["split"] == split)
        if len(seqs) != spec["expected_sequences"]:
            raise RuntimeError(f"{split}: {len(seqs)} sequences, expected {spec['expected_sequences']}")
        stats: Counter = Counter()
        split_gt = SplitGT(DATASET, split, "uavdt", [], drop_coverage=UAVDT_DROP_COVERAGE)
        mot_seqs = [convert_sequence(s, split, raw, out, attrs, stats, split_gt) for s in seqs]
        if split == "test":
            for s in mot_seqs:
                expected = cfg["expected_test_lengths"][s.name]
                if s.length != expected:
                    raise RuntimeError(f"{s.name}: {s.length} frames, toolkit expects {expected}")
        split_gt.save(out / "det" / "gt" / f"{split}.json")
        bench = write_benchmark(out / "mot", split, mot_seqs, targets,
                                threshold=UAVDT_DROP_COVERAGE, agnostic="vehicle")
        report[split] = {**stats, **bench}
        (out / "mot" / split / "stats.json").write_text(json.dumps(report[split], indent=2))
        print(f"uavdt/{split}: {len(seqs)} sequences, {stats['frames']} frames, "
              f"{stats['det_objects']} DET boxes, {stats['mot_rows_without_class']} MOT rows "
              f"without a class in gt_whole")
    names = {i: n for i, n in enumerate(targets)}
    import yaml  # noqa: PLC0415

    (out / "det" / "dataset.yaml").write_text(yaml.safe_dump(
        {"path": str((out / "det").resolve()), "train": "images/train", "val": "images/test",
         "names": names}, sort_keys=False))
    return report
