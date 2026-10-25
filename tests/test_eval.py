import numpy as np
import pytest

from aerotrack.config import apply_overrides
from aerotrack.data.classes import get_class_map
from aerotrack.data.gt import ImageGT, SplitGT, to_coco
from aerotrack.data.ignore import coverage, ignore_mask, keep_outside, paint_ignored
from aerotrack.eval.detection import coco_metrics, prediction_class_map
from aerotrack.eval.latency import summarize


# -- ignore rule ------------------------------------------------------------------------
def test_coverage_uses_union_of_regions():
    # Two adjacent ignore boxes each cover half of the box: per-box overlap would say 0.5
    # each, the union says 1.0 (this is why the official rule uses the union).
    mask = ignore_mask(np.array([[0, 0, 10, 20], [10, 0, 10, 20]]), 40, 40)
    assert coverage(np.array([[0, 0, 20, 20]]), mask)[0] == pytest.approx(1.0)
    assert coverage(np.array([[10, 0, 20, 20]]), mask)[0] == pytest.approx(0.5)


def test_keep_outside_thresholds():
    mask = ignore_mask(np.array([[0, 0, 10, 10]]), 40, 40)
    boxes = np.array([[0, 0, 10, 10], [5, 0, 10, 10], [20, 20, 5, 5]])
    assert keep_outside(boxes, mask, 0.5).tolist() == [False, False, True]  # VisDrone
    assert keep_outside(boxes, mask, 1.0).tolist() == [False, True, True]  # UAVDT
    assert keep_outside(boxes, None).all()


def test_paint_preserves_labelled_pixels():
    img = np.zeros((20, 20, 3), np.uint8)
    paint_ignored(img, np.array([[0, 0, 20, 20]]), np.array([[5, 5, 5, 5]]))
    assert (img[5:10, 5:10] == 0).all()
    assert (img[0:5] == 114).all()


# -- detection metrics ------------------------------------------------------------------
def _split(boxes, classes, ignore=()):
    return SplitGT("t", "val", "visdrone", [ImageGT(
        id=1, file="x.jpg", width=200, height=200, boxes=np.array(boxes, float),
        classes=np.array(classes), ignore=np.array(ignore, float).reshape(-1, 4))])


def _dets(rows):
    return [{"image_id": 1, "category_id": c, "bbox": list(b), "score": s} for b, c, s in rows]


def test_perfect_predictions_score_one():
    gt = _split([[10, 10, 50, 50], [100, 100, 20, 20]], [4, 1])
    coco, _ = to_coco(gt, get_class_map("visdrone"))
    ids = {c["name"]: c["id"] for c in coco["categories"]}
    assert min(a["id"] for a in coco["annotations"]) == 1  # pycocotools treats 0 as unmatched
    m = coco_metrics(coco, _dets([([10, 10, 50, 50], ids["car"], 0.9),
                                  ([100, 100, 20, 20], ids["pedestrian"], 0.8)]))
    assert m["mAP50-95"] == pytest.approx(1.0)
    assert m["per_class"]["car"]["AP"] == pytest.approx(1.0)


def test_false_positive_lowers_ap():
    gt = _split([[10, 10, 50, 50]], [4])
    coco, _ = to_coco(gt, get_class_map("visdrone"))
    car = next(c["id"] for c in coco["categories"] if c["name"] == "car")
    m = coco_metrics(coco, _dets([([120, 120, 50, 50], car, 0.95), ([10, 10, 50, 50], car, 0.5)]))
    assert m["mAP50"] == pytest.approx(0.5, abs=0.02)


def test_excluded_class_is_neither_tp_nor_fp():
    # A van (excluded in the COCO baseline map) detected as a car must not be an FP.
    gt = _split([[10, 10, 50, 50], [100, 100, 40, 40]], [4, 5])
    coco, masks = to_coco(gt, get_class_map("visdrone_to_coco"))
    assert len(coco["annotations"]) == 1
    car = next(c["id"] for c in coco["categories"] if c["name"] == "car")
    van_det = np.array([[100, 100, 40, 40]])
    assert not keep_outside(van_det, masks[1], gt.drop_coverage)[0]
    m = coco_metrics(coco, _dets([([10, 10, 50, 50], car, 0.5)]))
    assert m["mAP50-95"] == pytest.approx(1.0)


def test_ground_truth_inside_ignored_region_is_dropped():
    gt = _split([[10, 10, 20, 20], [150, 150, 20, 20]], [4, 4], ignore=[[0, 0, 60, 60]])
    coco, _ = to_coco(gt, get_class_map("visdrone"))
    assert len(coco["annotations"]) == 1


def test_prediction_mapping_is_by_name():
    cm = get_class_map("visdrone_to_coco")
    coco_names = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck",
                  8: "boat"}
    mapping = prediction_class_map(coco_names, cm)
    assert 8 not in mapping
    assert mapping[0] == cm.targets.index("person") + 1
    with pytest.raises(ValueError):
        prediction_class_map({0: "person"}, cm)  # targets without a model class


# -- latency ----------------------------------------------------------------------------
def test_latency_total_is_per_iteration():
    samples = [{"preprocess": 1, "inference": 10, "postprocess": 1}] * 9 + \
              [{"preprocess": 1, "inference": 100, "postprocess": 50}]
    s = summarize(samples)
    assert s["total"]["median_ms"] == pytest.approx(12)
    assert s["total"]["max_ms"] == pytest.approx(151)
    assert s["fps_median"] == pytest.approx(1000 / 12)
    assert s["total"]["p95_ms"] > s["inference"]["p95_ms"] + s["preprocess"]["p95_ms"]


# -- config / runtime -------------------------------------------------------------------
def test_overrides_parse_yaml_values():
    cfg = apply_overrides({"a": {"b": 1}}, ["a.b=3", "a.c=[1, 2]", "d=true"])
    assert cfg == {"a": {"b": 3, "c": [1, 2]}, "d": True}


def test_device_resolution(monkeypatch):
    import torch

    from aerotrack.runtime import resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu"
    with pytest.raises(RuntimeError):
        resolve_device("cuda")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("auto") == "mps"


def test_run_record_is_complete(tmp_path):
    import json

    from aerotrack.experiments import start_run

    with start_run("t", "test", {"x": 1}, seed=7, device="cpu") as run:
        run.summary({"metric": 0.5})
    rec = json.loads((run.dir / "run.json").read_text())
    assert rec["seed"] == 7 and rec["device"] == "cpu" and rec["config"] == {"x": 1}
    assert rec["git"]["commit"] is None or len(rec["git"]["commit"]) == 40
    assert rec["status"] == "finished" and rec["summary"]["metric"] == 0.5
    assert {"torch", "onnxruntime", "tensorrt", "jetson_power_mode"} <= set(rec["env"])


@pytest.mark.network
def test_detector_matches_ultralytics_predict():
    """Our pipeline vs Ultralytics' own predict on a stock image: same letterbox family and
    NMS, so the top detections must agree."""
    import cv2
    from ultralytics.utils import ASSETS

    from aerotrack.inference import Detector
    from aerotrack.ultra import load_yolo

    img = cv2.imread(str(ASSETS / "bus.jpg"))
    ours = Detector("yolo26n.pt", device="cpu", conf=0.5)(img)
    ref = load_yolo("yolo26n.pt").predict(img, conf=0.5, device="cpu", imgsz=640, verbose=False)[0]
    assert len(ours) == len(ref.boxes)
    assert sorted(ours[:, 5].astype(int)) == sorted(ref.boxes.cls.int().tolist())
