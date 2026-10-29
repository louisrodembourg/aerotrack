"""ByteTrack (Zhang et al., ECCV 2022) on top of aerotrack.track.kalman.

Per frame:
  1. Kalman-predict every tracked and lost track.
  2. Match high-score detections (>= high) to them on IoU, the cost fused with the
     detection score (1 - IoU * score), Hungarian assignment, cost <= match_thresh.
  3. Match low-score detections (low < score < high) to the still-unmatched TRACKED
     tracks on plain IoU (cost <= 0.5): the "byte" step that recovers occluded or blurred
     objects instead of discarding their weak detections. Unmatched tracked -> lost.
  4. Unconfirmed tracks (born last frame) take leftover high detections (cost <= 0.7)
     or are removed.
  5. Leftover high detections >= new_thresh start tracks; tracks lost for more than
     `buffer` frames are removed; tracked/lost duplicates (IoU > 0.85) keep the older.

Thresholds and bookkeeping follow Ultralytics' BYTETracker (8.4.187) step for step, so
the two can be compared on identical detections (tests/test_tracking.py). Association is
class-agnostic, as there: a track keeps its id when its detection's class flips (VisDrone
car/van confusions) and reports the class of its latest detection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from aerotrack.track.kalman import KalmanBoxFilter, xyah_to_xyxy, xyxy_to_xyah

TRACKED, LOST, REMOVED = 1, 2, 3


@dataclass
class Track:
    mean: np.ndarray
    cov: np.ndarray
    score: float
    cls: int
    start_frame: int
    frame_id: int
    track_id: int = 0
    state: int = TRACKED
    activated: bool = False

    @property
    def xyxy(self) -> np.ndarray:
        return xyah_to_xyxy(self.mean[:4])


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-7)


def assign(cost: np.ndarray, thresh: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Minimum-cost one-to-one matching, then pairs costlier than `thresh` are dropped.
    (Ultralytics' default solver is lap.lapjv with cost_limit; its scipy path, used here,
    thresholds after solving. See DECISIONS.md.)"""
    if not cost.size:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    rows, cols = linear_sum_assignment(cost)
    pairs = [(r, c) for r, c in zip(rows, cols, strict=True) if cost[r, c] <= thresh]
    mr, mc = {r for r, _ in pairs}, {c for _, c in pairs}
    return (pairs, [r for r in range(cost.shape[0]) if r not in mr],
            [c for c in range(cost.shape[1]) if c not in mc])


class ByteTrack:
    def __init__(self, high: float = 0.25, low: float = 0.1, new: float = 0.25,
                 buffer: int = 30, match: float = 0.8, fuse_score: bool = True) -> None:
        self.high, self.low, self.new, self.buffer = high, low, new, buffer
        self.match, self.fuse_score = match, fuse_score
        self.kf = KalmanBoxFilter()
        self.tracked: list[Track] = []
        self.lost: list[Track] = []
        self.frame_id = 0
        self._next_id = 1

    # -- helpers ----------------------------------------------------------------------
    def _new(self, det: np.ndarray) -> Track:
        mean, cov = self.kf.initiate(xyxy_to_xyah(det[:4]))
        return Track(mean, cov, float(det[4]), int(det[5]), self.frame_id, self.frame_id)

    def _correct(self, t: Track, det: np.ndarray) -> None:
        t.mean, t.cov = self.kf.update(t.mean, t.cov, xyxy_to_xyah(det[:4]))
        t.score, t.cls, t.frame_id = float(det[4]), int(det[5]), self.frame_id
        t.state, t.activated = TRACKED, True

    def _cost(self, tracks: list[Track], dets: np.ndarray, fuse: bool) -> np.ndarray:
        iou = iou_matrix(np.array([t.xyxy for t in tracks]).reshape(-1, 4), dets[:, :4])
        return 1 - (iou * dets[None, :, 4] if fuse and iou.size else iou)

    # -- one frame --------------------------------------------------------------------
    def update(self, dets: np.ndarray) -> np.ndarray:
        """dets (N, 6): x1, y1, x2, y2, score, class. Returns the active tracks as
        (M, 7): x1, y1, x2, y2 (Kalman-corrected), track id, score, class."""
        self.frame_id += 1
        dets = np.asarray(dets, dtype=np.float64).reshape(-1, 6)
        valid = (dets[:, 2] > dets[:, 0]) & (dets[:, 3] > dets[:, 1])
        high = dets[valid & (dets[:, 4] >= self.high)]
        low = dets[valid & (dets[:, 4] > self.low) & (dets[:, 4] < self.high)]

        unconfirmed = [t for t in self.tracked if not t.activated]
        pool = [t for t in self.tracked if t.activated] + self.lost
        if pool:
            mean = np.array([t.mean for t in pool])
            mean[[i for i, t in enumerate(pool) if t.state != TRACKED], 7] = 0  # lost: freeze h
            mean, cov = self.kf.predict(mean, np.array([t.cov for t in pool]))
            for t, m, c in zip(pool, mean, cov, strict=True):
                t.mean, t.cov = m, c

        updated, refound, lost, removed = [], [], [], []

        def apply(t: Track, det: np.ndarray) -> None:
            (updated if t.state == TRACKED else refound).append(t)
            self._correct(t, det)

        pairs, u_track, u_high = assign(self._cost(pool, high, self.fuse_score), self.match)
        for i, j in pairs:
            apply(pool[i], high[j])

        remaining = [pool[i] for i in u_track if pool[i].state == TRACKED]
        pairs, u_rem, _ = assign(self._cost(remaining, low, False), 0.5)
        for i, j in pairs:
            apply(remaining[i], low[j])
        for i in u_rem:
            remaining[i].state = LOST
            lost.append(remaining[i])

        high = high[u_high]
        pairs, u_unc, u_high = assign(self._cost(unconfirmed, high, self.fuse_score), 0.7)
        for i, j in pairs:
            self._correct(unconfirmed[i], high[j])
            updated.append(unconfirmed[i])
        for i in u_unc:
            unconfirmed[i].state = REMOVED
            removed.append(unconfirmed[i])

        for j in u_high:
            if high[j, 4] >= self.new:
                t = self._new(high[j])
                t.track_id, self._next_id = self._next_id, self._next_id + 1
                t.activated = self.frame_id == 1  # otherwise confirmed by a second match
                updated.append(t)

        for t in self.lost:
            if self.frame_id - t.frame_id > self.buffer:
                t.state = REMOVED
                removed.append(t)

        self._merge(updated, refound, lost, removed)
        return np.array([[*t.xyxy, t.track_id, t.score, t.cls]
                         for t in self.tracked if t.activated]).reshape(-1, 7)

    def predict_only(self) -> np.ndarray:
        """A frame without detections (hybrid keyframe mode): every track is carried one
        frame forward by its Kalman prediction and reported there; no track changes
        state, and the frame still counts towards the lost-track buffer."""
        self.frame_id += 1
        pool = [t for t in self.tracked if t.activated] + self.lost
        if pool:
            mean = np.array([t.mean for t in pool])
            mean[[i for i, t in enumerate(pool) if t.state != TRACKED], 7] = 0
            mean, cov = self.kf.predict(mean, np.array([t.cov for t in pool]))
            for t, m, c in zip(pool, mean, cov, strict=True):
                t.mean, t.cov = m, c
        return np.array([[*t.xyxy, t.track_id, t.score, t.cls]
                         for t in self.tracked if t.activated]).reshape(-1, 7)

    def _merge(self, updated, refound, lost, removed) -> None:
        """End-of-frame pools, in the reference order (Ultralytics merge_track_pools)."""
        ids = lambda ts: {t.track_id for t in ts}  # noqa: E731
        tracked = [t for t in self.tracked if t.state == TRACKED]
        for extra in (updated, refound):
            tracked += [t for t in extra if t.track_id not in ids(tracked)]
        gone = ids(tracked) | ids(removed)
        self.lost = [t for t in self.lost if t.track_id not in gone] + \
            [t for t in lost if t.track_id not in ids(removed)]
        # A tracked and a lost track on the same object (IoU > 0.85): keep the older.
        dist = 1 - iou_matrix(np.array([t.xyxy for t in tracked]).reshape(-1, 4),
                              np.array([t.xyxy for t in self.lost]).reshape(-1, 4))
        drop_a, drop_b = set(), set()
        for p, q in zip(*np.where(dist < 0.15), strict=True):
            age_p = tracked[p].frame_id - tracked[p].start_frame
            age_q = self.lost[q].frame_id - self.lost[q].start_frame
            (drop_b.add(q) if age_p > age_q else drop_a.add(p))
        self.tracked = [t for i, t in enumerate(tracked) if i not in drop_a]
        self.lost = [t for i, t in enumerate(self.lost) if i not in drop_b]
