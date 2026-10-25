"""COCO-style detection metrics on the dataset-neutral ground truth.

- Ignored regions and excluded classes are handled with the dataset's official rule
  (aerotrack.data.ignore), on BOTH sides, before pycocotools sees anything.
- maxDets is raised to 500: VisDrone images routinely hold more than 100 objects, and the
  official VisDrone-DET protocol reports AP at 500 detections. pycocotools' `summarize()`
  hard-codes the 100-detection column, so metrics are read from the precision array.
- AP small/medium/large use the COCO area thresholds (32^2, 96^2) unchanged, for
  comparability with published numbers; most VisDrone objects fall in "small".
"""

from __future__ import annotations

import contextlib
import io
from collections.abc import Callable, Iterable

import cv2
import numpy as np

from aerotrack.data.classes import ClassMap
from aerotrack.data.gt import SplitGT, to_coco
from aerotrack.data.ignore import keep_outside
from aerotrack.paths import from_data_relative

AREAS = ("all", "small", "medium", "large")


def prediction_class_map(model_names: dict[int, str], cmap: ClassMap) -> dict[int, int]:
    """Model class id -> COCO category id of the evaluation, matched BY NAME. Classes of the
    model with no counterpart (e.g. COCO `boat`) are dropped; the mapping actually used is
    written into the report, so it can be checked."""
    cat_ids = {name: i + 1 for i, name in enumerate(cmap.targets)}
    mapping = {int(k): cat_ids[v] for k, v in model_names.items() if v in cat_ids}
    if not mapping:
        raise ValueError(f"no model class matches the targets of {cmap.name!r}: {cmap.targets}")
    missing = set(cat_ids) - {model_names[k] for k in mapping}
    if missing:
        raise ValueError(f"targets {missing} of {cmap.name!r} have no model class")
    return mapping


def predictions_to_coco(
    gt: SplitGT,
    masks: dict,
    detect: Callable[[np.ndarray], np.ndarray],
    class_mapping: dict[int, int],
    progress: Callable[[int, int], None] | None = None,
) -> list[dict]:
    dets = []
    for i, im in enumerate(gt.images):
        img = cv2.imread(str(from_data_relative(im.file)))
        if img is None:
            raise FileNotFoundError(from_data_relative(im.file))
        out = detect(img)
        cls = out[:, 5].astype(int)
        known = np.isin(cls, list(class_mapping))
        out, cls = out[known], cls[known]
        xywh = np.column_stack([out[:, 0], out[:, 1], out[:, 2] - out[:, 0], out[:, 3] - out[:, 1]])
        keep = keep_outside(xywh, masks[im.id], gt.drop_coverage)
        for box, score, c in zip(xywh[keep], out[keep, 4], cls[keep], strict=True):
            dets.append({"image_id": im.id, "category_id": class_mapping[int(c)],
                         "bbox": [float(v) for v in box], "score": float(score)})
        if progress:
            progress(i + 1, len(gt.images))
    return dets


def coco_metrics(coco_gt: dict, dets: list[dict], max_det: int = 500) -> dict:
    from pycocotools.coco import COCO  # noqa: PLC0415
    from pycocotools.cocoeval import COCOeval  # noqa: PLC0415

    names = {c["id"]: c["name"] for c in coco_gt["categories"]}
    n_gt = {n: sum(a["category_id"] == i for a in coco_gt["annotations"]) for i, n in names.items()}
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = coco_gt
        gt.createIndex()
        if not dets:
            return _empty(names, n_gt)
        dt = gt.loadRes(dets)
        ev = COCOeval(gt, dt, iouType="bbox")
        ev.params.maxDets = [1, 10, 100, max_det]
        ev.evaluate()
        ev.accumulate()

    precision = ev.eval["precision"]  # [T iou, R recall, K class, A area, M maxdet]
    recall = ev.eval["recall"]  # [T, K, A, M]
    iou_thrs = ev.params.iouThrs

    def ap(area: str = "all", iou: float | None = None, k: int | None = None) -> float:
        p = precision[..., AREAS.index(area), -1]
        if iou is not None:
            p = p[np.flatnonzero(np.isclose(iou_thrs, iou))]
        if k is not None:
            p = p[:, :, k]
        valid = p[p > -1]
        return float(valid.mean()) if valid.size else float("nan")

    r = recall[:, :, 0, -1]
    per_class = {}
    for k, cid in enumerate(ev.params.catIds):
        per_class[names[cid]] = {"AP": ap(k=k), "AP50": ap(iou=0.5, k=k), "n_gt": n_gt[names[cid]]}
    return {
        "mAP50-95": ap(),
        "mAP50": ap(iou=0.5),
        "mAP75": ap(iou=0.75),
        "AP_small": ap("small"),
        "AP_medium": ap("medium"),
        "AP_large": ap("large"),
        f"AR{max_det}": float(r[r > -1].mean()) if (r > -1).any() else float("nan"),
        "max_det": max_det,
        "n_images": len(coco_gt["images"]),
        "n_gt": int(sum(n_gt.values())),
        "n_dets": len(dets),
        "per_class": per_class,
    }


def _empty(names: dict, n_gt: dict) -> dict:
    zero = {k: 0.0 for k in ("mAP50-95", "mAP50", "mAP75", "AP_small", "AP_medium", "AP_large")}
    return {**zero, "n_dets": 0, "n_gt": int(sum(n_gt.values())),
            "per_class": {n: {"AP": 0.0, "AP50": 0.0, "n_gt": n_gt[n]} for n in names.values()}}


def evaluate_detection(
    gt: SplitGT, cmap: ClassMap, detect: Callable[[np.ndarray], np.ndarray],
    model_names: dict[int, str], max_det: int,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    coco_gt, masks = to_coco(gt, cmap)
    mapping = prediction_class_map(model_names, cmap)
    dets = predictions_to_coco(gt, masks, detect, mapping, progress)
    metrics = coco_metrics(coco_gt, dets, max_det=max_det)
    metrics["class_map"] = cmap.name
    metrics["model_class_to_target"] = {model_names[k]: cmap.targets[v - 1]
                                        for k, v in sorted(mapping.items())}
    metrics["excluded_classes"] = {cmap.vocabulary[k]: why for k, why in cmap.excluded.items()}
    return metrics


def iter_images(gt: SplitGT) -> Iterable[np.ndarray]:
    for im in gt.images:
        yield cv2.imread(str(from_data_relative(im.file)))
