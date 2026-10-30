"""Copy-paste of small objects of rare classes, for box-only detection data.

Ultralytics' CopyPaste only copies segmentation masks and does nothing on box labels
(`if len(labels["instances"].segments) == 0: return labels`), so VisDrone needs its own.
A bank of small objects of the rare classes is cropped once from the training images;
each training image then receives, with probability `p`, up to `max_paste` of them, at
the scale of the receiving image, on free background: not overlapping any labelled box,
already pasted object or ignored region. Rectangular crops, no blending (box labels give
no object outline; DECISIONS.md).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np


def rare_classes(counts: dict[int, int], share: float) -> list[int]:
    """Classes holding less than `share` of all training instances."""
    total = sum(counts.values())
    return sorted(c for c, n in counts.items() if n < share * total)


@dataclass
class Bank:
    crops: list[np.ndarray]   # BGR pixels at the source image's original resolution
    cls: np.ndarray           # (N,)

    def __len__(self) -> int:
        return len(self.crops)


def build_bank(labels: list[dict], classes: list[int], min_side: int, max_side: int,
               workers: int = 4) -> Bank:
    """Crop every labelled object of `classes` whose longer side, in original pixels, is
    within [min_side, max_side]. `labels` are Ultralytics label dicts (im_file, shape
    (h, w), normalised xywh bboxes, cls)."""
    wanted = set(classes)

    def crops_of(lb: dict) -> list[tuple[np.ndarray, int]]:
        h, w = lb["shape"]
        b, c = lb["bboxes"], lb["cls"].reshape(-1).astype(int)
        sel = [k for k in range(len(c)) if c[k] in wanted
               and min_side <= max(b[k, 2] * w, b[k, 3] * h) <= max_side]
        if not sel:
            return []
        img = cv2.imread(lb["im_file"])
        out = []
        for k in sel:
            x0, y0 = int(round((b[k, 0] - b[k, 2] / 2) * w)), int(round((b[k, 1] - b[k, 3] / 2) * h))
            x1, y1 = int(round((b[k, 0] + b[k, 2] / 2) * w)), int(round((b[k, 1] + b[k, 3] / 2) * h))
            crop = img[max(y0, 0):y1, max(x0, 0):x1]
            if crop.size:
                out.append((crop.copy(), int(c[k])))
        return out

    with ThreadPoolExecutor(workers) as ex:
        found = [x for xs in ex.map(crops_of, labels) for x in xs]
    return Bank([x[0] for x in found], np.array([x[1] for x in found], dtype=int))


def _overlaps(box: np.ndarray, others: np.ndarray) -> bool:
    if not len(others):
        return False
    return bool(((box[0] < others[:, 2]) & (others[:, 0] < box[2]) &
                 (box[1] < others[:, 3]) & (others[:, 1] < box[3])).any())


def paste(img: np.ndarray, boxes_xyxy: np.ndarray, ignore_xyxy: np.ndarray, bank: Bank,
          scale: float, n: int, rng: np.random.Generator, tries: int = 20
          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Paste up to `n` bank objects into `img` (modified in place: pass a copy). Classes are
    drawn uniformly, then an instance of the class, so the rarest class is not drowned by
    the less rare ones. Returns the image, the pasted boxes (xyxy pixels) and classes."""
    h, w = img.shape[:2]
    taken = np.concatenate([boxes_xyxy.reshape(-1, 4), ignore_xyxy.reshape(-1, 4)])
    by_class = {c: np.flatnonzero(bank.cls == c) for c in np.unique(bank.cls)}
    keys = sorted(by_class)
    new_boxes, new_cls = [], []
    for _ in range(n):
        c = keys[rng.integers(len(keys))]
        crop = bank.crops[by_class[c][rng.integers(len(by_class[c]))]]
        ch, cw = max(1, round(crop.shape[0] * scale)), max(1, round(crop.shape[1] * scale))
        if ch >= h or cw >= w:
            continue
        for _ in range(tries):
            x0, y0 = int(rng.integers(0, w - cw)), int(rng.integers(0, h - ch))
            box = np.array([x0, y0, x0 + cw, y0 + ch], dtype=float)
            if not _overlaps(box, taken):
                img[y0:y0 + ch, x0:x0 + cw] = cv2.resize(crop, (cw, ch)) if scale != 1 else crop
                taken = np.vstack([taken, box])
                new_boxes.append(box)
                new_cls.append(c)
                break
    return img, np.array(new_boxes).reshape(-1, 4), np.array(new_cls, dtype=int)
