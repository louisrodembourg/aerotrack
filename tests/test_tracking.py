"""Tracker checks: the Kalman filter on known motion, and ByteTrack against Ultralytics'
BYTETracker on identical detections."""

import sys
import types

import numpy as np
import pytest

from aerotrack.track.bytetrack import ByteTrack
from aerotrack.track.kalman import KalmanBoxFilter, xyah_to_xyxy, xyxy_to_xyah


def test_kalman_learns_constant_velocity():
    kf = KalmanBoxFilter()
    box = np.array([100.0, 50.0, 140.0, 130.0])
    mean, cov = kf.initiate(xyxy_to_xyah(box))
    for _ in range(30):
        box = box + [3.0, -1.0, 3.0, -1.0]
        m, c = kf.predict(mean[None], cov[None])
        mean, cov = kf.update(m[0], c[0], xyxy_to_xyah(box))
    assert mean[4:6] == pytest.approx([3.0, -1.0], abs=0.05)
    m, _ = kf.predict(mean[None], cov[None])  # one step ahead, no measurement
    assert xyah_to_xyxy(m[0, :4]) == pytest.approx(box + [3, -1, 3, -1], abs=0.5)
    assert np.allclose(cov, cov.T) and (np.linalg.eigvalsh(cov) > 0).all()


def _scene(seed: int, frames: int = 60, n: int = 12) -> list[np.ndarray]:
    """Boxes moving at constant speed with jitter, missed detections, low-score frames,
    class flips and clutter: every branch of ByteTrack gets exercised."""
    rng = np.random.default_rng(seed)
    start = rng.uniform(50, 900, (n, 2))
    size = rng.uniform(10, 60, (n, 2))
    vel = rng.uniform(-4, 4, (n, 2))
    cls = rng.integers(0, 3, n)
    out = []
    for f in range(frames):
        xy = start + vel * f + rng.normal(0, 1.0, (n, 2))
        dets = np.c_[xy, xy + size, rng.uniform(0.3, 0.95, n), cls]
        dets[rng.random(n) < 0.15, 4] = rng.uniform(0.11, 0.24)  # occluded: weak score
        dets[rng.random(n) < 0.05, 5] = (cls[0] + 1) % 3          # class flip
        keep = rng.random(n) > 0.1                                # missed detections
        clutter = np.c_[rng.uniform(0, 900, (2, 2)), rng.uniform(0, 900, (2, 2)) + 20,
                        rng.uniform(0.05, 0.4, 2), rng.integers(0, 3, 2)]
        out.append(np.r_[dets[keep], clutter])
    return out


@pytest.fixture
def reference_tracker(monkeypatch):
    # Ultralytics' matching module requires `lap` at import (and would pip-install it).
    # Its own scipy path is used instead, the solver our implementation also uses.
    monkeypatch.setitem(sys.modules, "lap", types.SimpleNamespace(__version__="stub"))
    from ultralytics.trackers import byte_tracker
    from ultralytics.trackers.utils import matching

    solve = matching.linear_assignment
    monkeypatch.setattr(matching, "linear_assignment",
                        lambda cost, thresh, use_lap=True: solve(cost, thresh, use_lap=False))
    args = types.SimpleNamespace(track_high_thresh=0.25, track_low_thresh=0.1,
                                 new_track_thresh=0.25, track_buffer=30, match_thresh=0.8,
                                 fuse_score=True)
    return byte_tracker.BYTETracker(args)


class _Boxes:
    def __init__(self, d: np.ndarray):
        self.d = d
        self.conf, self.cls, self.xyxy = d[:, 4], d[:, 5], d[:, :4]
        self.xywh = np.c_[(d[:, :2] + d[:, 2:4]) / 2, d[:, 2:4] - d[:, :2]]

    def __getitem__(self, m):
        return _Boxes(self.d[m])

    def __len__(self):
        return len(self.d)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_bytetrack_matches_ultralytics(reference_tracker, seed):
    """Same tracks, same Kalman boxes, same classes, frame by frame. Ids are compared up to
    a renaming fixed for the whole sequence: when several tracks start in one frame, the
    reference numbers them in the iteration order of a Python frozenset (its scipy path),
    an implementation artefact; ours numbers them in detection order."""
    ours = ByteTrack()
    rename: dict[int, int] = {}
    for dets in _scene(seed):
        a = ours.update(dets)
        b = np.asarray(reference_tracker.update(_Boxes(dets))).reshape(-1, 8)[:, :7]
        assert len(a) == len(b)
        a, b = a[np.lexsort(a[:, :2].T)], b[np.lexsort(b[:, :2].T)]  # pair rows by box
        assert a[:, :4] == pytest.approx(b[:, :4], abs=1e-3)
        assert a[:, 6].tolist() == b[:, 6].tolist()
        for ia, ib in zip(a[:, 4].astype(int), b[:, 4].astype(int), strict=True):
            assert rename.setdefault(ia, ib) == ib
    assert len(set(rename.values())) == len(rename)  # a bijection


def test_write_tracks_drops_ignored_regions_per_frame(tmp_path):
    from aerotrack.eval.tracking import write_tracks

    ignore = tmp_path / "seq.txt"
    ignore.write_text("1,0,0,100,100\n")  # frame 1 only: top-left 100x100 is ignored
    rows = np.array([[1, 1, 10, 10, 50, 50, 0.9, 0],    # frame 1, inside: dropped
                     [1, 2, 200, 200, 240, 240, 0.9, 0],  # frame 1, outside: kept
                     [2, 1, 10, 10, 50, 50, 0.9, 0],    # frame 2, no ignore: kept
                     [2, 3, 300, 300, 330, 330, 0.8, 1]])
    seq = {"name": "seq", "width": 400, "height": 400}
    write_tracks(rows, seq, ignore, {0: "car", 1: "van"}, tmp_path / "out")
    car = (tmp_path / "out" / "car" / "data" / "seq.txt").read_text().splitlines()
    assert [ln.split(",")[:2] for ln in car] == [["1", "2"], ["2", "1"]]
    assert car[0].split(",")[2:6] == ["200.00", "200.00", "40.00", "40.00"]  # x, y, w, h
    assert len((tmp_path / "out" / "van" / "data" / "seq.txt").read_text().splitlines()) == 1


def test_seen_by_detector_matches_the_source_video(tmp_path):
    from aerotrack.eval.tracking import seen_by_detector

    for split, name in (("train", "0000201_00412_d_0000045.jpg"),
                        ("test-dev", "0000073_00600_d_0000001.jpg")):
        (tmp_path / "images" / split).mkdir(parents=True)
        (tmp_path / "images" / split / name).touch()
    (tmp_path / "images" / "val").mkdir()
    assert seen_by_detector("uav0000201_00000_v", tmp_path) == ["train/0000201_00412_d_0000045.jpg"]
    assert seen_by_detector("uav0000073_00600_v", tmp_path) == []  # DET test-dev: never trained on
    assert seen_by_detector("uav0000020_00000_v", tmp_path) == []  # 0000020_ is not 0000201_
