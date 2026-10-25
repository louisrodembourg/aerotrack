"""Dataset-neutral detection ground truth.

Converters write one JSON per split in the source vocabulary, with ignored regions kept
separately. Turning that into a COCO evaluation set happens later, per class map, so the
same converted data serves the VisDrone-trained models, the COCO baseline and UAVDT
transfer without re-conversion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from aerotrack.data.classes import ClassMap
from aerotrack.data.ignore import DROP_COVERAGE, ignore_mask, keep_outside


@dataclass
class ImageGT:
    id: int
    file: str  # relative to DATA_DIR
    width: int
    height: int
    boxes: np.ndarray  # (N, 4) xywh pixels, source vocabulary
    classes: np.ndarray  # (N,) source class ids
    ignore: np.ndarray  # (M, 4) xywh pixels
    extra: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "file": self.file,
            "width": self.width,
            "height": self.height,
            "boxes": [[*map(float, b), int(c)] for b, c in zip(self.boxes, self.classes, strict=True)],
            "ignore": [list(map(float, b)) for b in self.ignore],
            **({"extra": self.extra} if self.extra else {}),
        }

    @classmethod
    def from_json(cls, d: dict) -> ImageGT:
        rows = np.asarray(d["boxes"], dtype=float).reshape(-1, 5)
        return cls(
            id=d["id"],
            file=d["file"],
            width=d["width"],
            height=d["height"],
            boxes=rows[:, :4],
            classes=rows[:, 4].astype(int),
            ignore=np.asarray(d["ignore"], dtype=float).reshape(-1, 4),
            extra=d.get("extra", {}),
        )


@dataclass
class SplitGT:
    dataset: str
    split: str
    vocabulary: str
    images: list[ImageGT]
    # Official ignore rule of the dataset (see aerotrack.data.ignore).
    drop_coverage: float = DROP_COVERAGE

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset": self.dataset,
            "split": self.split,
            "vocabulary": self.vocabulary,
            "drop_coverage": self.drop_coverage,
            "images": [im.to_json() for im in self.images],
        }
        path.write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: Path) -> SplitGT:
        d = json.loads(Path(path).read_text())
        return cls(d["dataset"], d["split"], d["vocabulary"],
                   [ImageGT.from_json(x) for x in d["images"]], d["drop_coverage"])

    def subset(self, n: int | None, seed: int = 0) -> SplitGT:
        if n is None or n >= len(self.images):
            return self
        rng = np.random.default_rng(seed)
        idx = sorted(rng.choice(len(self.images), size=n, replace=False))
        return SplitGT(self.dataset, self.split, self.vocabulary,
                       [self.images[i] for i in idx], self.drop_coverage)


def evaluation_mask(im: ImageGT, cmap: ClassMap) -> np.ndarray | None:
    """Ignored regions plus every object of an excluded class: a model is neither rewarded
    nor penalised for what the class map deliberately leaves out (e.g. a COCO model calling
    a VisDrone `van` a car)."""
    excluded = np.isin(im.classes, list(cmap.excluded))
    regions = np.concatenate([im.ignore, im.boxes[excluded]]) if len(im.boxes) else im.ignore
    if len(regions) == 0:
        return None
    return ignore_mask(regions, im.height, im.width)


def to_coco(gt: SplitGT, cmap: ClassMap) -> tuple[dict, dict[int, np.ndarray | None]]:
    """COCO ground truth for `cmap`, plus the per-image ignore masks needed to filter
    predictions with the same rule. Annotation and image ids start at 1: pycocotools
    uses 0 to mean "unmatched" and would silently corrupt matching otherwise."""
    if gt.vocabulary != cmap.source:
        raise ValueError(f"ground truth is in {gt.vocabulary!r}, class map expects {cmap.source!r}")
    cat_ids = {name: i + 1 for i, name in enumerate(cmap.targets)}
    images, annotations, masks = [], [], {}
    ann_id = 1
    for im in gt.images:
        images.append({"id": im.id, "file_name": im.file, "width": im.width, "height": im.height})
        mask = evaluation_mask(im, cmap)
        masks[im.id] = mask
        mapped = np.isin(im.classes, list(cmap.mapping))
        keep = mapped & keep_outside(im.boxes, mask, gt.drop_coverage)
        for box, cls in zip(im.boxes[keep], im.classes[keep], strict=True):
            x, y, w, h = map(float, box)
            annotations.append({
                "id": ann_id, "image_id": im.id, "category_id": cat_ids[cmap.mapping[int(cls)]],
                "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0,
            })
            ann_id += 1
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i, "name": n} for n, i in cat_ids.items()],
    }
    return coco, masks
