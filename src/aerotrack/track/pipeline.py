"""One video, frame by frame: detector(s) -> tracker. The same object serves accuracy
(`step`) and latency (`timed_step`), as `inference.Detector` does for detection.

Hybrid keyframe mode (`interval` K > 1): the key detector runs on one frame in K. In
between, either a light detector runs (`light`), or no detector at all and the tracker
only predicts (its Kalman filter carries the tracks forward). The cost per frame is then
uneven, so throughput is judged on the mean frame time, with p95 and max reported.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from aerotrack.inference import STAGES

KINDS = ("key", "light", "predict")


class TrackingPipeline:
    def __init__(self, key: Callable, make_tracker: Callable, keep: set[int],
                 interval: int = 1, light: Callable | None = None) -> None:
        if interval < 1:
            raise ValueError("interval must be >= 1")
        self.key, self.light, self.interval = key, light, interval
        self.make_tracker, self.keep = make_tracker, np.array(sorted(keep))
        self.reset()

    def reset(self) -> None:
        """Start of a sequence: new tracker, frame count from zero (first frame is key)."""
        self.tracker = self.make_tracker()
        self.i = 0

    def _kind(self) -> str:
        if self.i % self.interval == 0:
            return "key"
        return "light" if self.light is not None else "predict"

    def _track(self, det: np.ndarray | None) -> np.ndarray:
        if det is None:
            return self.tracker.predict_only()
        return self.tracker.update(det[np.isin(det[:, 5].astype(int), self.keep)])

    def step(self, img) -> np.ndarray:
        """Tracks (M, 7): x1, y1, x2, y2, id, score, class."""
        kind = self._kind()
        det = None if kind == "predict" else (self.key if kind == "key" else self.light)(img)
        self.i += 1
        return self._track(det)

    def timed_step(self, img) -> tuple[np.ndarray, dict]:
        """Same as step, with per-stage ms (0 for a detector stage that did not run)."""
        kind = self._kind()
        if kind == "predict":
            det, t = None, dict.fromkeys(STAGES, 0.0)
        else:
            det, t = (self.key if kind == "key" else self.light).timed(img)
        self.i += 1
        t0 = time.perf_counter()
        tracks = self._track(det)
        t["tracking"] = (time.perf_counter() - t0) * 1e3
        t["kind"] = kind
        return tracks, t
