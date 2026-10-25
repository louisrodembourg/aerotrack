"""VisDrone2019-MOT -> TrackEval MOTChallenge benchmarks (one per evaluated class).

Raw annotation row (VisDrone2018-MOT-toolkit README):
    frame_index, target_id, bbox_left, bbox_top, bbox_width, bbox_height,
    score, object_category, truncation, occlusion

Evaluated classes and ignore handling follow the official toolkit: the five classes of
the `visdrone_mot` class map, and boxes at least half inside the union of ignored regions
(category 0 or 11) dropped. Rows with score 0 are added to the ignored regions, as in the
DET converter.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from aerotrack.data.classes import get_class_map
from aerotrack.data.mot import SequenceGT, write_benchmark
from aerotrack.data.visdrone_det import parse_annotation
from aerotrack.paths import PROCESSED_DIR, RAW_DIR

DATASET = "visdrone_mot"


def load_sequence(raw_split_dir: Path, name: str, stats: Counter) -> SequenceGT:
    cmap = get_class_map("visdrone_mot")
    targets = cmap.targets
    image_dir = raw_split_dir / "sequences" / name
    frames = sorted(image_dir.glob("*.jpg"))
    with Image.open(frames[0]) as im:
        width, height = im.size

    rows = parse_annotation(raw_split_dir / "annotations" / f"{name}.txt", ncols=10)
    frame, tid, xywh = rows[:, 0], rows[:, 1], rows[:, 2:6]
    score, cat = rows[:, 6], rows[:, 7]
    if frame.min() < 1 or frame.max() > len(frames):
        raise ValueError(f"{name}: frame indices {frame.min()}..{frame.max()} vs {len(frames)} images")

    is_region = np.isin(cat, list(cmap.ignore))
    is_flagged = ~is_region & (score == 0)
    is_object = ~is_region & ~is_flagged

    target_idx = np.array(
        [targets.index(cmap.mapping[c]) if c in cmap.mapping else -1 for c in cat], dtype=np.int64
    )
    objects = np.column_stack([frame, tid, xywh, target_idx])[is_object].astype(float)
    ignore = np.column_stack([frame, xywh])[is_region | is_flagged].astype(float)

    # A track whose class changes over time would land in two benchmarks with one id.
    obj_cat = cat[is_object]
    obj_tid = tid[is_object]
    multi = sum(len(set(obj_cat[obj_tid == t])) > 1 for t in np.unique(obj_tid))

    stats["rows"] += len(rows)
    stats["objects"] += int(is_object.sum())
    stats["objects_not_evaluated_class"] += int((target_idx[is_object] < 0).sum())
    stats["ignore_rows"] += int((is_region | is_flagged).sum())
    stats["score0_objects_to_ignore"] += int(is_flagged.sum())
    stats["tracks_with_class_change"] += int(multi)
    stats["frames"] += len(frames)
    return SequenceGT(name, image_dir, len(frames), width, height, ".jpg", objects, ignore)


def convert(cfg: dict, raw_root: Path | None = None, out_dir: Path | None = None) -> dict:
    raw_root = raw_root or RAW_DIR / DATASET
    out_dir = out_dir or PROCESSED_DIR / DATASET
    targets = get_class_map("visdrone_mot").targets
    report = {}
    for split, spec in cfg["splits"].items():
        raw_split = raw_root / spec["dir"]
        if not raw_split.exists():
            if spec.get("optional"):
                print(f"visdrone_mot/{split}: not downloaded, skipped")
                continue
            raise FileNotFoundError(raw_split)
        names = sorted(p.name for p in (raw_split / "sequences").iterdir() if p.is_dir())
        stats: Counter = Counter()
        seqs = [load_sequence(raw_split, n, stats) for n in names]
        bench = write_benchmark(out_dir, split, seqs, targets)
        expected = spec.get("expected_sequences")
        if expected is not None and len(seqs) != expected:
            raise RuntimeError(f"{split}: {len(seqs)} sequences, README announces {expected}")
        report[split] = {**stats, **bench}
        (out_dir / split / "stats.json").write_text(json.dumps(report[split], indent=2))
        print(f"visdrone_mot/{split}: {len(seqs)} sequences, {stats['frames']} frames, "
              f"{bench['boxes_written']} evaluated boxes")
    return report
