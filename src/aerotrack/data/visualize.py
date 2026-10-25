"""Draw converted annotations on random frames.

Everything is read back from the CONVERTED files (YOLO labels, TrackEval gt.txt, ignore
files), not from the raw annotations: the point is to check what training and evaluation
will actually consume. Output: reports/viz/<dataset>/*.jpg (git-ignored: dataset images
are under a non-commercial licence).
"""

from __future__ import annotations

import colorsys
from pathlib import Path

import cv2
import numpy as np

from aerotrack.data.classes import get_class_map
from aerotrack.data.ignore import paint_ignored
from aerotrack.paths import PROCESSED_DIR, REPORTS_DIR

TRAIL = 30  # frames of trajectory drawn behind each track: an id switch shows as a jump


def _color(i: int) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb((i * 0.618034) % 1.0, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def _label(img: np.ndarray, text: str, x: int, y: int, color) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    y = max(th + 2, y)
    cv2.rectangle(img, (x, y - th - 3), (x + tw + 2, y), color, -1)
    cv2.putText(img, text, (x + 1, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)


def _draw_ignore(img: np.ndarray, regions: np.ndarray) -> None:
    for x, y, w, h in np.asarray(regions).reshape(-1, 4).astype(int):
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.line(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.line(img, (x + w, y), (x, y + h), (0, 0, 255), 1)


def _header(img: np.ndarray, text: str) -> np.ndarray:
    bar = np.full((28, img.shape[1], 3), 32, dtype=np.uint8)
    cv2.putText(bar, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def _read_rows(path: Path, ncols: int) -> np.ndarray:
    text = path.read_text().strip() if path.exists() else ""
    if not text:
        return np.zeros((0, ncols))
    return np.loadtxt(text.splitlines(), delimiter=",", ndmin=2)[:, :ncols]


def viz_yolo(root: Path, split: str, names: dict[int, str], n: int, seed: int, out: Path,
             title: str) -> list[Path]:
    """YOLO labels (de-normalised) on the image as the training loader sees it, ignored
    regions painted grey and outlined in red."""
    images = sorted((root / "images" / split).glob("*.jpg"))
    rng = np.random.default_rng(seed)
    written = []
    for idx in rng.choice(len(images), size=min(n, len(images)), replace=False):
        img_path = images[idx]
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        label_file = root / "labels" / split / f"{img_path.stem}.txt"
        labels = np.loadtxt(label_file, ndmin=2) if label_file.read_text().strip() else np.zeros((0, 5))
        boxes = np.column_stack([
            (labels[:, 1] - labels[:, 3] / 2) * w, (labels[:, 2] - labels[:, 4] / 2) * h,
            labels[:, 3] * w, labels[:, 4] * h,
        ]) if len(labels) else np.zeros((0, 4))
        ign_file = root / "ignore" / split / f"{img_path.stem}.txt"
        ignore = np.loadtxt(ign_file, ndmin=2) if ign_file.exists() and ign_file.read_text().strip() \
            else np.zeros((0, 4))
        paint_ignored(img, ignore, boxes)
        _draw_ignore(img, ignore)
        for (x, y, bw, bh), c in zip(boxes.astype(int), labels[:, 0].astype(int), strict=True):
            cv2.rectangle(img, (x, y), (x + bw, y + bh), _color(c), 2)
        for c in sorted(set(labels[:, 0].astype(int).tolist())):
            _label(img, names[c], 8, 22 + 20 * sorted(set(labels[:, 0].astype(int))).index(c),
                   _color(c))
        img = _header(img, f"{title} {split}/{img_path.name}  objects={len(labels)} "
                           f"ignore_regions={len(ignore)} (red X, grey = masked for training)")
        dest = out / f"{split}_{img_path.stem}.jpg"
        cv2.imwrite(str(dest), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        written.append(dest)
    return written


def viz_mot(root: Path, split: str, classes: list[str], n: int, seed: int, out: Path,
            title: str) -> list[Path]:
    """TrackEval gt.txt of every class, with ids and a trail of each track's past centres."""
    split_dir = root / split
    seqs = [s for s in (split_dir / "seqmap.txt").read_text().split()[1:]]
    rng = np.random.default_rng(seed)
    written = []
    for k in range(n):
        seq = seqs[k % len(seqs)] if n >= len(seqs) else seqs[rng.integers(len(seqs))]
        info = dict(line.split("=", 1) for line in
                    (split_dir / classes[0] / seq / "seqinfo.ini").read_text().splitlines()
                    if "=" in line and not line.startswith(";"))
        length = int(info["seqLength"])
        frame = int(rng.integers(TRAIL + 1, length + 1)) if length > TRAIL + 1 else length
        im_dir = (split_dir / classes[0] / seq / info["imDir"]).resolve()
        frames = sorted(im_dir.glob(f"*{info['imExt']}"))
        img = cv2.imread(str(frames[frame - 1]))
        n_boxes = 0
        for ci, cls in enumerate(classes):
            gt = _read_rows(split_dir / cls / seq / "gt" / "gt.txt", 6)
            if not len(gt):
                continue
            now = gt[gt[:, 0] == frame]
            n_boxes += len(now)
            for _, tid, x, y, w, h in now:
                past = gt[(gt[:, 1] == tid) & (gt[:, 0] > frame - TRAIL) & (gt[:, 0] <= frame)]
                past = past[np.argsort(past[:, 0])]
                pts = np.column_stack([past[:, 2] + past[:, 4] / 2, past[:, 3] + past[:, 5] / 2])
                cv2.polylines(img, [pts.astype(np.int32)], False, _color(int(tid)), 1)
                cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), _color(ci), 2)
                _label(img, f"{cls[:3]}{int(tid)}", int(x), int(y) - 1, _color(ci))
        ign = _read_rows(split_dir / "ignore" / f"{seq}.txt", 5)
        _draw_ignore(img, ign[ign[:, 0] == frame, 1:5] if len(ign) else ign)
        img = _header(img, f"{title} {split}/{seq} frame {frame}/{length}  evaluated boxes={n_boxes} "
                           f"(label = class+track id, line = last {TRAIL} frames)")
        dest = out / f"{split}_{seq}_{frame:06d}.jpg"
        cv2.imwrite(str(dest), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        written.append(dest)
    return written


def visualize(dataset: str, n: int = 12, seed: int = 0) -> Path:
    out = REPORTS_DIR / "viz" / dataset
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.jpg"):
        old.unlink()
    root = PROCESSED_DIR / dataset
    if dataset == "visdrone_det":
        names = {sid - 1: name for sid, name in get_class_map("visdrone").mapping.items()}
        for split in ("train", "val"):
            viz_yolo(root, split, names, n // 2, seed, out, "VisDrone-DET")
    elif dataset == "visdrone_mot":
        classes = get_class_map("visdrone_mot").targets
        viz_mot(root, "val", classes, n, seed, out, "VisDrone-MOT")
    elif dataset == "uavdt":
        classes = get_class_map("uavdt").targets
        viz_mot(root / "mot", "test", classes, n // 2, seed, out, "UAVDT-MOT")
        names = dict(enumerate(classes))
        viz_yolo(root / "det", "test", names, n // 2, seed, out, "UAVDT-DET")
    else:
        raise KeyError(dataset)
    return out
