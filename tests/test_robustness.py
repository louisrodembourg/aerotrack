"""Week 11: VisDrone -> UAVDT class transfer, attribute groups, synthetic degradations."""

import json

import numpy as np
import pytest

from aerotrack.data.classes import get_class_map
from aerotrack.data.gt import ImageGT, SplitGT
from aerotrack.eval.degrade import KINDS, degrade
from aerotrack.eval.detection import evaluate_detection, prediction_class_map

VISDRONE_NAMES = {0: "pedestrian", 1: "people", 2: "bicycle", 3: "car", 4: "van",
                  5: "truck", 6: "tricycle", 7: "awning-tricycle", 8: "bus", 9: "motor"}


def _uavdt(files):
    """UAVDT-like ground truth: one car per frame, frames named after their sequence."""
    return SplitGT("uavdt", "test", "uavdt", [ImageGT(
        id=i + 1, file=f, width=200, height=200, boxes=np.array([[10.0, 10, 50, 40]]),
        classes=np.array([1]), ignore=np.zeros((0, 4))) for i, f in enumerate(files)],
        drop_coverage=1.0)


def test_visdrone_vans_score_as_uavdt_cars():
    m = prediction_class_map(VISDRONE_NAMES, get_class_map("uavdt_to_visdrone"))
    cats = {n: i + 1 for i, n in enumerate(get_class_map("uavdt_to_visdrone").targets)}
    assert m[3] == m[4] == cats["car"]  # car, van
    assert m[5] == cats["truck"] and m[8] == cats["bus"]
    assert 0 not in m  # pedestrians: no UAVDT class
    agnostic = prediction_class_map({2: "car", 7: "truck", 5: "bus", 0: "person"},
                                    get_class_map("uavdt_vehicle"))  # COCO names
    assert set(agnostic) == {2, 7, 5} and len(set(agnostic.values())) == 1


def test_a_car_and_a_van_box_on_one_vehicle_count_once(monkeypatch, tmp_path):
    """Without merging, the van duplicate would be a false positive of `car`."""
    import cv2

    img = tmp_path / "M0101" / "img000001.jpg"
    img.parent.mkdir()
    cv2.imwrite(str(img), np.zeros((200, 200, 3), np.uint8))
    monkeypatch.setattr("aerotrack.eval.detection.from_data_relative", lambda p: tmp_path / p)
    gt = _uavdt(["M0101/img000001.jpg"])

    def detect(_):  # the same vehicle as car (0.9) and as van (0.6), slightly offset
        return np.array([[10, 10, 60, 50, 0.9, 3], [11, 10, 61, 50, 0.6, 4]], np.float32)

    m = evaluate_detection(gt, get_class_map("uavdt_to_visdrone"), detect, VISDRONE_NAMES, 500)
    assert m["n_dets"] == 1
    assert m["mAP50"] == pytest.approx(1.0)


def test_attribute_groups_follow_each_frames_sequence(tmp_path):
    from aerotrack.eval.harness import attribute_groups

    attrs = tmp_path / "attributes.json"
    attrs.write_text(json.dumps({
        "M0101": {"split": "test", "weather": ["night"], "view": ["front", "side"],
                  "long_term": False},
        "M0202": {"split": "test", "weather": ["daylight"], "view": ["bird"], "long_term": True},
    }))
    gt = _uavdt(["raw/uavdt/M0101/img1.jpg", "raw/uavdt/M0101/img2.jpg",
                 "raw/uavdt/M0202/img1.jpg"])
    g = attribute_groups(gt, attrs)
    assert g["weather=night"] == {1, 2} and g["weather=daylight"] == {3}
    assert g["view=front"] == g["view=side"] == {1, 2}  # several views: in each group
    assert g["long_term=true"] == {3}
    assert "split=test" not in g


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_degradations_are_deterministic_and_change_the_frame(kind):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
    level = {"motion_blur": 11, "noise": 15, "jpeg": 20, "low_light": 0.3}[kind]
    a, b = degrade(img, kind, level), degrade(img.copy(), kind, level)
    assert a.shape == img.shape and a.dtype == np.uint8
    np.testing.assert_array_equal(a, b)  # same frame, same degradation, any run
    assert np.abs(a.astype(int) - img.astype(int)).mean() > 1
    if kind == "low_light":
        assert a.mean() < 0.5 * img.mean()


def test_group_scores_equal_a_separate_evaluation_of_the_group():
    """Groups reuse one per-image matching; that must equal scoring the group alone."""
    from aerotrack.eval.detection import coco_metrics, subset_coco

    rng = np.random.default_rng(0)
    images, anns, dets = [], [], []
    for i in range(1, 31):
        images.append({"id": i, "width": 400, "height": 400})
        for _ in range(rng.integers(1, 6)):
            x, y, w, h = *rng.uniform(0, 300, 2), *rng.uniform(8, 80, 2)
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": 1,
                         "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0})
            if rng.random() < 0.8:  # a jittered true positive
                dets.append({"image_id": i, "category_id": 1, "score": float(rng.random()),
                             "bbox": [x + rng.normal(0, 4), y + rng.normal(0, 4), w, h]})
        for _ in range(rng.integers(0, 4)):  # false positives
            dets.append({"image_id": i, "category_id": 1, "score": float(rng.random()),
                         "bbox": [*rng.uniform(0, 300, 2), *rng.uniform(8, 80, 2)]})
    coco = {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "vehicle"}]}
    groups = {"a": set(range(1, 11)), "b": set(range(5, 31, 3))}
    shared = coco_metrics(coco, dets, groups=groups)["groups"]
    for label, ids in groups.items():
        alone = coco_metrics(*subset_coco(coco, dets, ids))
        for k in ("mAP50-95", "mAP50", "mAP70", "AP_small", "n_gt", "n_images", "n_dets"):
            assert shared[label][k] == pytest.approx(alone[k], nan_ok=True), (label, k)
