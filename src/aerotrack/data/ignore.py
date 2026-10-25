"""Ignored-region handling, shared by training (masking) and evaluation (exclusion).

Evaluation follows the official VisDrone toolkits (`dropObjects.m`): a box is dropped,
on both the ground-truth and the prediction side, when at least half of its area lies in
the union of ignored regions. Using the union matters: VisDrone often tiles a crowded
area with several adjacent ignore boxes, and a per-box overlap test (COCO `iscrowd`)
would keep detections that straddle two of them.
"""

from __future__ import annotations

import numpy as np

DROP_COVERAGE = 0.5  # VisDrone dropObjects.m: `igrVal/(h*w) < 0.5` keeps the box
# UAVDT CalculateDetectionPR_*.m only drops a detection strictly inside an ignore box.
# Each dataset keeps its official rule; the threshold travels with the converted ground
# truth (SplitGT.drop_coverage) so evaluation cannot pick the wrong one.
UAVDT_DROP_COVERAGE = 1.0


def ignore_mask(ignore_xywh: np.ndarray, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for x, y, w, h in np.asarray(ignore_xywh, dtype=float).reshape(-1, 4):
        x0, y0 = max(0, int(round(x))), max(0, int(round(y)))
        x1, y1 = min(width, int(round(x + w))), min(height, int(round(y + h)))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
    return mask


def coverage(boxes_xywh: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fraction of each box's area covered by `mask`, via an integral image (O(1)/box)."""
    boxes = np.asarray(boxes_xywh, dtype=float).reshape(-1, 4)
    if len(boxes) == 0:
        return np.zeros(0)
    h, w = mask.shape
    integral = np.zeros((h + 1, w + 1), dtype=np.int64)
    integral[1:, 1:] = mask.astype(np.int64).cumsum(0).cumsum(1)
    x0 = np.clip(np.round(boxes[:, 0]), 0, w).astype(int)
    y0 = np.clip(np.round(boxes[:, 1]), 0, h).astype(int)
    x1 = np.clip(np.round(boxes[:, 0] + boxes[:, 2]), 0, w).astype(int)
    y1 = np.clip(np.round(boxes[:, 1] + boxes[:, 3]), 0, h).astype(int)
    covered = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
    area = np.maximum(boxes[:, 2] * boxes[:, 3], 1e-9)
    return covered / area


def keep_outside(boxes_xywh: np.ndarray, mask: np.ndarray | None,
                 threshold: float = DROP_COVERAGE) -> np.ndarray:
    """Boolean keep-mask: True for boxes covered by less than `threshold` of ignored area."""
    boxes = np.asarray(boxes_xywh, dtype=float).reshape(-1, 4)
    if mask is None or not mask.any():
        return np.ones(len(boxes), dtype=bool)
    # The epsilon makes threshold=1.0 mean "fully inside" despite float rounding.
    return coverage(boxes, mask) < threshold - 1e-9


def paint_ignored(
    image: np.ndarray,
    ignore_xywh: np.ndarray,
    keep_xywh: np.ndarray,
    fill: int = 114,
) -> np.ndarray:
    """Training-time masking, in place. Ignored pixels are painted with the letterbox grey
    so the network sees them as padding rather than as unlabeled background; pixels that
    belong to a labeled box are preserved, otherwise an ignore box overlapping a real car
    would erase part of a positive."""
    h, w = image.shape[:2]
    mask = ignore_mask(ignore_xywh, h, w)
    if not mask.any():
        return image
    mask &= ~ignore_mask(keep_xywh, h, w)
    image[mask] = fill
    return image
