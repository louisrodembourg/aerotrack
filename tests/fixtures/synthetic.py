"""Synthetic raw datasets in the exact on-disk formats of VisDrone and UAVDT.

Generators live here and never in src/: nothing synthetic may be mistaken for real data.
Every file is written under a pytest tmp_path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

W, H = 320, 240


def _image(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (H, W, 3), dtype=np.uint8)).save(path)


# Rows: left, top, width, height, score, category, truncation, occlusion
DET_ROWS = {
    "img_a": [
        "10,20,30,40,1,4,0,0",        # car
        "100,50,20,60,1,1,0,1",       # pedestrian
        "200,100,50,50,0,0,0,0",      # ignored region (category 0)
        "250,10,40,40,1,11,0,0",      # others (category 11) -> ignored
        "300,200,40,60,1,9,1,0,",     # bus crossing the right/bottom border, trailing comma
        "330,10,10,10,1,6,0,0",       # truck fully outside the image -> degenerate
    ],
    "img_b": [
        "5,5,10,10,1,10,0,0",         # motor
    ],
}


def make_visdrone_det(root: Path, split_dir: str = "VisDrone2019-DET-val") -> Path:
    base = root / split_dir
    for i, (stem, rows) in enumerate(DET_ROWS.items()):
        _image(base / "images" / f"{stem}.jpg", seed=i)
        ann = base / "annotations" / f"{stem}.txt"
        ann.parent.mkdir(parents=True, exist_ok=True)
        ann.write_text("\n".join(rows) + "\n")
    return base


def make_visdrone_mot(root: Path, n_frames: int = 6) -> Path:
    """One sequence: a car moving right (id 1), a pedestrian (id 2), a person of class
    `people` (id 3, not evaluated), an ignored region covering a van (id 4) on frame 3."""
    base = root / "VisDrone2019-MOT-val"
    seq = "uav_seq"
    for f in range(1, n_frames + 1):
        _image(base / "sequences" / seq / f"{f:07d}.jpg", seed=f)
    rows = []
    for f in range(1, n_frames + 1):
        rows.append(f"{f},1,{10 + 5 * f},20,30,20,1,4,0,0")
        rows.append(f"{f},2,100,{100 + 2 * f},10,25,1,1,0,0")
        rows.append(f"{f},3,150,150,10,20,1,2,0,0")
    rows.append("3,4,200,50,40,30,1,5,0,0")
    rows.append("3,0,190,40,70,60,0,0,-1,-1")
    ann = base / "annotations" / f"{seq}.txt"
    ann.parent.mkdir(parents=True, exist_ok=True)
    ann.write_text("\n".join(rows) + "\n")
    return base
