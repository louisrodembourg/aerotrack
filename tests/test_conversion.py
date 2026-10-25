import hashlib
from pathlib import Path

import numpy as np
import pytest

from aerotrack.data.classes import all_class_maps, get_class_map, vocabulary
from aerotrack.data.gt import SplitGT
from aerotrack.data.mot import SequenceGT, write_benchmark
from aerotrack.data.visdrone_det import convert_split, parse_annotation
from aerotrack.data.visdrone_mot import load_sequence
from tests.fixtures.synthetic import DET_ROWS, H, W, make_visdrone_det, make_visdrone_mot


def _hashes(root: Path) -> dict[str, str]:
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


# -- class maps -------------------------------------------------------------------------
@pytest.mark.parametrize("name", all_class_maps())
def test_every_source_class_is_accounted_for(name):
    get_class_map(name)  # validate() raises on any unaccounted or doubly-assigned id


def test_baseline_map_excludes_ambiguous_classes():
    cm = get_class_map("visdrone_to_coco")
    assert set(cm.excluded) == {5, 7, 8}  # van, tricycle, awning-tricycle
    assert cm.targets == ["person", "bicycle", "car", "truck", "bus", "motorcycle"]
    assert vocabulary("visdrone")[0] == "ignored-regions"


# -- VisDrone-DET -----------------------------------------------------------------------
@pytest.fixture
def det(tmp_path, isolated_paths):
    raw = make_visdrone_det(isolated_paths / "raw" / "visdrone_det")
    before = _hashes(raw)
    out = isolated_paths / "processed" / "visdrone_det"
    stats = convert_split(raw, out, "val")
    return raw, out, stats, before


def test_det_object_accounting(det):
    _, _, stats, _ = det
    n_rows = sum(len(r) for r in DET_ROWS.values())
    assert stats["rows"] == n_rows
    ignore_rows = stats["ignored_regions_cat0"] + stats["others_cat11"] + stats["score0_objects_to_ignore"]
    assert stats["objects_kept"] + stats["objects_degenerate_dropped"] + ignore_rows == n_rows
    assert stats["objects_degenerate_dropped"] == 1
    assert stats["objects_clipped"] == 1


def test_det_yolo_labels_are_normalised_and_complete(det):
    _, out, stats, _ = det
    lines = [ln for f in (out / "labels" / "val").glob("*.txt") for ln in f.read_text().splitlines()]
    assert len(lines) == stats["objects_kept"]
    arr = np.array([[float(x) for x in ln.split()] for ln in lines])
    assert set(arr[:, 0].astype(int)) <= set(range(10))
    cx, cy, w, h = arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
    assert np.all((cx - w / 2 >= -1e-6) & (cx + w / 2 <= 1 + 1e-6))
    assert np.all((cy - h / 2 >= -1e-6) & (cy + h / 2 <= 1 + 1e-6))
    assert np.all((w > 0) & (h > 0))


def test_det_class_ids_shift_by_one(det):
    _, out, _, _ = det
    first = (out / "labels" / "val" / "img_a.txt").read_text().splitlines()[0].split()
    assert first[0] == "3"  # VisDrone car (4) -> YOLO 3
    x, y, w, h = 10, 20, 30, 40
    assert float(first[1]) == pytest.approx((x + w / 2) / W, abs=1e-6)
    assert float(first[2]) == pytest.approx((y + h / 2) / H, abs=1e-6)


def test_det_ignore_regions_kept_separately(det):
    _, out, _, _ = det
    ign = np.loadtxt(out / "ignore" / "val" / "img_a.txt", ndmin=2)
    assert len(ign) == 2  # category 0 + category 11
    gt = SplitGT.load(out / "gt" / "val.json")
    assert len(gt.images[0].ignore) == 2
    assert 0 not in gt.images[0].classes and 11 not in gt.images[0].classes


def test_det_raw_data_untouched_and_linked(det):
    raw, out, _, before = det
    assert _hashes(raw) == before
    link = out / "images" / "val" / "img_a.jpg"
    assert link.is_symlink()
    assert link.resolve() == (raw / "images" / "img_a.jpg").resolve()


def test_parse_rejects_short_rows(tmp_path):
    f = tmp_path / "bad.txt"
    f.write_text("1,2,3\n")
    with pytest.raises(ValueError):
        parse_annotation(f)


# -- VisDrone-MOT -----------------------------------------------------------------------
@pytest.fixture
def mot(isolated_paths):
    from collections import Counter

    raw = make_visdrone_mot(isolated_paths / "raw" / "visdrone_mot")
    stats = Counter()
    seq = load_sequence(raw, "uav_seq", stats)
    out = isolated_paths / "processed" / "visdrone_mot"
    bench = write_benchmark(out, "val", [seq], get_class_map("visdrone_mot").targets)
    return seq, out / "val", stats, bench


def _gt(split_dir: Path, cls: str) -> np.ndarray:
    text = (split_dir / cls / "uav_seq" / "gt" / "gt.txt").read_text().strip()
    return np.loadtxt(text.splitlines(), delimiter=",", ndmin=2) if text else np.zeros((0, 9))


def test_mot_object_accounting(mot):
    seq, _, stats, bench = mot
    not_evaluated = stats["objects_not_evaluated_class"]
    dropped = sum(c["dropped_in_ignore"] for c in bench["per_class"].values())
    assert bench["boxes_written"] + dropped + not_evaluated == stats["objects"]
    assert dropped == 1  # the van under the ignored region on frame 3


def test_mot_track_ids_consistent(mot):
    _, split_dir, _, _ = mot
    car, ped = _gt(split_dir, "car"), _gt(split_dir, "pedestrian")
    assert set(car[:, 1]) == {1} and set(ped[:, 1]) == {2}
    for gt in (car, ped):
        pairs = {(int(f), int(t)) for f, t in gt[:, :2]}
        assert len(pairs) == len(gt)  # one box per (frame, id)
        assert np.all(np.diff(gt[:, 0]) >= 0)  # sorted by frame
        assert gt[:, 0].min() >= 1 and gt[:, 0].max() <= 6
    assert np.allclose(np.diff(car[:, 2]), 5)  # the same object, moving 5 px per frame


def test_mot_classes_relabelled_for_trackeval(mot):
    _, split_dir, _, _ = mot
    car = _gt(split_dir, "car")
    assert np.all(car[:, 6:9] == 1)  # conf flag, class 1 (pedestrian slot), visibility


def test_mot_seqinfo_points_to_raw_images(mot):
    seq, split_dir, _, _ = mot
    ini = (split_dir / "car" / "uav_seq" / "seqinfo.ini").read_text()
    info = dict(ln.split("=", 1) for ln in ini.splitlines() if "=" in ln)
    assert int(info["seqLength"]) == 6
    assert (split_dir / "car" / "uav_seq" / info["imDir"]).resolve() == seq.image_dir.resolve()
    assert "frameRate" not in info


def test_mot_ground_truth_scores_perfectly_in_trackeval(mot):
    from aerotrack.eval.tracking import gt_self_check

    _, split_dir, _, _ = mot
    hota = gt_self_check("visdrone_mot", "val", root=split_dir.parent)
    assert set(hota) == {"car", "pedestrian"}  # empty classes are skipped
    assert all(v == pytest.approx(1.0) for v in hota.values())


def test_mot_threshold_one_keeps_partially_covered(tmp_path):
    objects = np.array([[1, 1, 0, 0, 10, 10, 0], [1, 2, 50, 50, 10, 10, 0]], dtype=float)
    ignore = np.array([[1, 5, 0, 20, 20], [1, 50, 50, 10, 10]], dtype=float)
    seq = SequenceGT("s", tmp_path, 1, 100, 100, ".jpg", objects, ignore)
    b = write_benchmark(tmp_path / "o", "x", [seq], ["car"], threshold=1.0)
    # id 1 is half covered (kept under the UAVDT rule), id 2 fully covered (dropped)
    assert b["per_class"]["car"]["boxes"] == 1


# -- UAVDT ------------------------------------------------------------------------------
def test_uavdt_attributes_parse_stray_space(tmp_path):
    from aerotrack.data.uavdt import read_attributes

    (tmp_path / "train").mkdir()
    (tmp_path / "test").mkdir()
    (tmp_path / "test" / "M0701 _attr.txt").write_text("0,0,1,0,0,1,0,1,1,0")
    (tmp_path / "train" / "M0101_attr.txt").write_text("1,0,0,0,1,0,1,1,0,0")
    a = read_attributes(tmp_path)
    assert a["M0701"] == {"split": "test", "weather": ["fog"], "altitude": ["high"],
                          "view": ["side", "bird"], "long_term": False}
    assert a["M0101"]["view"] == ["front", "side"]
