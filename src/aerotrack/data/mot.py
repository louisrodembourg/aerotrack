"""Shared writer for TrackEval's MOTChallenge format.

TrackEval's `MotChallenge2DBox` only evaluates the class `pedestrian` (id 1) and rejects
any other class id in tracker files. Multi-class tracking is therefore written as one
benchmark per class, each relabelling its own objects as class 1:

    <root>/<split>/<class>/<seq>/gt/gt.txt     frame,id,x,y,w,h,1,1,1
    <root>/<split>/<class>/<seq>/seqinfo.ini
    <root>/<split>/seqmap.txt                  shared by every class
    <root>/<split>/ignore/<seq>.txt            frame,x,y,w,h (pixels)

Evaluate with SKIP_SPLIT_FOL=True, GT_FOLDER=<root>/<split>/<class>,
SEQMAP_FILE=<root>/<split>/seqmap.txt. Tracker outputs must be filtered with the same
ignore regions (aerotrack.data.ignore.keep_outside) before TrackEval sees them, which is
what the official VisDrone toolkit does for both sides.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aerotrack.data.ignore import DROP_COVERAGE, ignore_mask, keep_outside


@dataclass
class SequenceGT:
    name: str
    image_dir: Path
    length: int
    width: int
    height: int
    image_ext: str
    # (N, 7): frame, track_id, x, y, w, h, target_class_index (-1 = not evaluated)
    objects: np.ndarray
    # (M, 5): frame, x, y, w, h
    ignore: np.ndarray


def drop_ignored(seq: SequenceGT, threshold: float = DROP_COVERAGE) -> np.ndarray:
    """Per-frame keep-mask over seq.objects (official rule, see aerotrack.data.ignore)."""
    keep = np.ones(len(seq.objects), dtype=bool)
    for frame in np.unique(seq.ignore[:, 0]).astype(int):
        regions = seq.ignore[seq.ignore[:, 0] == frame, 1:5]
        mask = ignore_mask(regions, seq.height, seq.width)
        rows = np.flatnonzero(seq.objects[:, 0] == frame)
        keep[rows] = keep_outside(seq.objects[rows, 2:6], mask, threshold)
    return keep


def write_seqinfo(path: Path, seq: SequenceGT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    im_dir = os.path.relpath(seq.image_dir, path.parent)
    path.write_text(
        "[Sequence]\n"
        f"name={seq.name}\n"
        f"imDir={im_dir}\n"
        "; frameRate is not published by the dataset and is deliberately not guessed\n"
        f"seqLength={seq.length}\n"
        f"imWidth={seq.width}\n"
        f"imHeight={seq.height}\n"
        f"imExt={seq.image_ext}\n"
    )


def write_benchmark(root: Path, split: str, sequences: list[SequenceGT], classes: list[str],
                    threshold: float = DROP_COVERAGE, agnostic: str | None = None) -> dict:
    """`agnostic`, when set, adds a benchmark of that name holding every object regardless
    of class (UAVDT's official MOT protocol is class-agnostic)."""
    split_dir = root / split
    stats: dict = {"sequences": len(sequences), "per_class": {}}
    (split_dir / "ignore").mkdir(parents=True, exist_ok=True)
    (split_dir / "seqmap.txt").write_text("name\n" + "".join(f"{s.name}\n" for s in sequences))

    kept_total = 0
    for seq in sequences:
        keep = drop_ignored(seq, threshold)
        ign = seq.ignore
        (split_dir / "ignore" / f"{seq.name}.txt").write_text(
            "".join(f"{int(f)},{x:g},{y:g},{w:g},{h:g}\n" for f, x, y, w, h in ign)
        )
        benchmarks = [*enumerate(classes)] + ([(None, agnostic)] if agnostic else [])
        for ci, cls in benchmarks:
            in_class = np.ones(len(seq.objects), bool) if ci is None else seq.objects[:, 6] == ci
            sel = keep & in_class
            rows = seq.objects[sel]
            rows = rows[np.lexsort((rows[:, 1], rows[:, 0]))]
            seq_dir = split_dir / cls / seq.name
            write_seqinfo(seq_dir / "seqinfo.ini", seq)
            gt = seq_dir / "gt" / "gt.txt"
            gt.parent.mkdir(parents=True, exist_ok=True)
            gt.write_text("".join(
                f"{int(f)},{int(t)},{x:g},{y:g},{w:g},{h:g},1,1,1\n" for f, t, x, y, w, h, _ in rows
            ))
            c = stats["per_class"].setdefault(cls, {"boxes": 0, "tracks": 0, "dropped_in_ignore": 0})
            c["boxes"] += len(rows)
            c["tracks"] += len(np.unique(rows[:, 1])) if len(rows) else 0
            c["dropped_in_ignore"] += int((~keep & in_class).sum())
            if ci is not None:
                kept_total += len(rows)
    stats["boxes_written"] = kept_total
    return stats
