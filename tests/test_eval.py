from pathlib import Path

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


def test_source_vocabulary_model_goes_through_the_ground_truth_map():
    # A VisDrone fine-tune scored on the COCO-mappable classes, like the zero-shot baseline.
    cm = get_class_map("visdrone_to_coco")
    visdrone_names = dict(enumerate(get_class_map("visdrone").targets))
    mapping = prediction_class_map(visdrone_names, cm)
    cat = {n: cm.targets.index(n) + 1 for n in cm.targets}
    by_name = {visdrone_names[k]: v for k, v in mapping.items()}
    assert by_name["pedestrian"] == by_name["people"] == cat["person"]
    assert by_name["motor"] == cat["motorcycle"]
    assert {"van", "tricycle", "awning-tricycle"}.isdisjoint(by_name)


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


def test_train_run_lookup_refuses_ambiguous_names():
    from aerotrack.experiments import find_train_run, start_run

    ids = []
    for _ in range(2):
        with start_run("ref", "train", {}, seed=0, device="cpu") as run:
            run.summary({"weights": f"{run.id}/best.pt"})
        ids.append(run.id)
    with pytest.raises(LookupError, match="2 finished training runs"):
        find_train_run("ref")
    assert find_train_run(ids[1])["summary"]["weights"] == f"{ids[1]}/best.pt"
    with pytest.raises(LookupError, match="no finished"):
        find_train_run("missing")


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


def test_latency_refused_on_a_throttled_mac(monkeypatch):
    from aerotrack.eval import harness

    def env(power):
        return lambda *a, **k: {"macos_power": power}

    monkeypatch.setattr(harness, "environment_info", env({"source": "battery", "low_power_mode": False}))
    with pytest.raises(SystemExit, match="throttled"):
        harness.check_power({"enabled": True})
    harness.check_power({"enabled": True, "allow_throttled": True})
    harness.check_power({"enabled": False})
    monkeypatch.setattr(harness, "environment_info", env({"source": "ac", "low_power_mode": True}))
    with pytest.raises(SystemExit):
        harness.check_power({})
    monkeypatch.setattr(harness, "environment_info", env({"source": "ac", "low_power_mode": False}))
    harness.check_power({})
    monkeypatch.setattr(harness, "environment_info", env(None))  # not a Mac
    harness.check_power({})


def test_onnx_cache_is_keyed_by_weights_content(tmp_path, monkeypatch):
    # Two fine-tunes are both called best.pt: they must not share an ONNX export.
    from aerotrack import inference

    class FakeModel:
        def __init__(self, path):
            self.path = path

        def export(self, **kw):
            out = tmp_path / "export.onnx"
            out.write_bytes(self.path.read_bytes())
            return str(out)

    monkeypatch.setattr(inference, "WEIGHTS_DIR", tmp_path / "w")
    monkeypatch.setattr(inference, "load_yolo", lambda p: FakeModel(Path(p)))
    paths = []
    for run, content in (("n", b"nano"), ("s", b"small")):
        best = tmp_path / run / "best.pt"
        best.parent.mkdir()
        best.write_bytes(content)
        paths.append(inference.export_onnx(best, 640, 500))
    assert paths[0] != paths[1]
    assert paths[1].read_bytes() == b"small"
    assert inference.export_onnx(tmp_path / "s" / "best.pt", 640, 500) == paths[1]


def test_resolve_weights_keeps_architectures_bare():
    from aerotrack import ultra
    assert ultra.resolve_weights("yolo26n-p2.yaml") == "yolo26n-p2.yaml"
    assert Path(ultra.resolve_weights("yolo26n.pt")) == ultra.WEIGHTS_DIR / "yolo26n.pt"


def test_tile_grid_covers_the_frame_with_overlap():
    from aerotrack.inference import Detector

    d = Detector.__new__(Detector)
    d.tiles, d.tile_overlap, d.tile_full = (2, 3), 0.2, True
    crops = d.crops(765, 1360)
    assert len(crops) == 7 and crops[-1] == (0, 0, 1360, 765)
    grid = crops[:-1]
    assert min(c[0] for c in grid) == 0 and max(c[2] for c in grid) == 1360
    assert min(c[1] for c in grid) == 0 and max(c[3] for c in grid) == 765
    assert grid[0][2] > grid[1][0] and grid[0][3] > grid[3][1]  # neighbours overlap


def test_merge_tiles_drops_truncated_boxes_keeps_other_classes():
    import torch

    from aerotrack.inference import merge_tiles

    det = torch.tensor([
        [100., 100., 200., 160., 0.9, 3.],  # full car, from the whole-frame pass
        [150., 100., 200., 160., 0.6, 3.],  # its right half, cut by a tile edge: IoU 0.5, IOS 1
        [150., 100., 200., 160., 0.5, 0.],  # same place, another class: kept
        [400., 400., 420., 430., 0.3, 3.],  # elsewhere: kept
    ])
    out = merge_tiles(det, ios=0.5, max_det=10)
    assert out[:, 4].tolist() == pytest.approx([0.9, 0.5, 0.3])


def test_tiled_detector_runs_one_batch_and_maps_boxes_to_the_frame():
    import cv2
    from ultralytics.utils import ASSETS

    from aerotrack.inference import Detector

    img = cv2.imread(str(ASSETS / "bus.jpg"))
    d = Detector("yolo26n.pt", device="cpu", conf=0.5, tiles=(2, 2))
    assert d.preprocess(img).shape[0] == 5
    det = d(img)
    h, w = img.shape[:2]
    assert len(det) and (det[:, [0, 2]] <= w + 1).all() and (det[:, [1, 3]] <= h + 1).all()
    assert 0 in det[:, 5].astype(int)  # the people in bus.jpg survive the merge
