"""Tracking on the per-class MOTChallenge benchmarks written by aerotrack.data.mot, scored
with TrackEval.

`track_sequence` runs detector -> tracker over one sequence's frames in order;
`write_tracks` filters the output with the sequence's ignore regions (the same rule the
ground truth went through) and writes one tracker file per class benchmark; `run_trackeval`
scores one class. `gt_self_check` scores the ground truth as a tracker (week 1).
"""

from __future__ import annotations

import configparser
import contextlib
import io
import shutil
import tempfile
import time
from collections.abc import Callable, Iterable
from pathlib import Path

import cv2
import numpy as np

from aerotrack.data.ignore import ignore_mask, keep_outside
from aerotrack.paths import PROCESSED_DIR

MOT_ROOTS = {"visdrone_mot": PROCESSED_DIR / "visdrone_mot", "uavdt": PROCESSED_DIR / "uavdt" / "mot"}


def benchmark_classes(split_dir: Path) -> list[str]:
    return sorted(p.name for p in split_dir.iterdir() if p.is_dir() and p.name != "ignore")


def _import_trackeval():
    """TrackEval (last commit 12c8791) still uses np.int / np.float / np.bool, removed in
    numpy 1.24, in its MOTChallenge loader and in the HOTA and Identity metrics. Those names
    were plain aliases of the builtins, so restoring them is behaviour-identical; pinning
    numpy < 1.24 is not an option (torch and Ultralytics need numpy 2)."""
    for alias, builtin in (("int", int), ("float", float), ("bool", bool)):
        if alias not in np.__dict__:
            setattr(np, alias, builtin)
    import trackeval  # noqa: PLC0415

    return trackeval


def run_trackeval(split_dir: Path, cls: str, trackers_dir: Path, tracker: str) -> dict:
    """HOTA / CLEAR / Identity for one class benchmark. Tracker files must live in
    <trackers_dir>/<tracker>/data/<seq>.txt (frame,id,x,y,w,h,conf,-1,-1,-1) and must
    already be filtered with the dataset's ignore rule."""
    trackeval = _import_trackeval()

    eval_cfg = {**trackeval.Evaluator.get_default_eval_config(),
                "USE_PARALLEL": False, "PRINT_RESULTS": False, "PRINT_CONFIG": False,
                "OUTPUT_SUMMARY": False, "OUTPUT_DETAILED": False, "PLOT_CURVES": False,
                "TIME_PROGRESS": False, "DISPLAY_LESS_PROGRESS": True}
    ds_cfg = {**trackeval.datasets.MotChallenge2DBox.get_default_dataset_config(),
              "GT_FOLDER": str(split_dir / cls), "TRACKERS_FOLDER": str(trackers_dir),
              "TRACKERS_TO_EVAL": [tracker], "SEQMAP_FILE": str(split_dir / "seqmap.txt"),
              "SKIP_SPLIT_FOL": True, "BENCHMARK": "aerotrack", "CLASSES_TO_EVAL": ["pedestrian"],
              "OUTPUT_FOLDER": str(trackers_dir / "_out")}
    quiet = {"PRINT_CONFIG": False}
    metrics = [trackeval.metrics.HOTA(quiet), trackeval.metrics.CLEAR(quiet),
               trackeval.metrics.Identity(quiet)]
    with contextlib.redirect_stdout(io.StringIO()):
        res, _ = trackeval.Evaluator(eval_cfg).evaluate(
            [trackeval.datasets.MotChallenge2DBox(ds_cfg)], metrics)
    r = res["MotChallenge2DBox"][tracker]["COMBINED_SEQ"]["pedestrian"]
    return {"HOTA": float(np.mean(r["HOTA"]["HOTA"])), "MOTA": float(r["CLEAR"]["MOTA"]),
            "IDF1": float(r["Identity"]["IDF1"]), "IDSW": int(r["CLEAR"]["IDSW"])}


def gt_self_check(dataset: str, split: str, root: Path | None = None) -> dict[str, float]:
    split_dir = (root or MOT_ROOTS[dataset]) / split
    seqs = (split_dir / "seqmap.txt").read_text().split()[1:]
    out = {}
    tmp = Path(tempfile.mkdtemp(prefix="aerotrack-trackeval-"))
    try:
        for cls in benchmark_classes(split_dir):
            data = tmp / cls / "gt" / "data"
            data.mkdir(parents=True)
            for seq in seqs:
                rows = (split_dir / cls / seq / "gt" / "gt.txt").read_text().splitlines()
                (data / f"{seq}.txt").write_text("".join(
                    ",".join(r.split(",")[:6]) + ",1,-1,-1,-1\n" for r in rows))
            if not any((split_dir / cls / s / "gt" / "gt.txt").stat().st_size for s in seqs):
                continue  # a class with no box at all has no defined HOTA
            out[cls] = run_trackeval(split_dir, cls, tmp / cls, "gt")["HOTA"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def sequences(split_dir: Path) -> list[dict]:
    """name, frame paths, size of each sequence of a split (from any class's seqinfo)."""
    cls = benchmark_classes(split_dir)[0]
    out = []
    for name in (split_dir / "seqmap.txt").read_text().split()[1:]:
        ini = split_dir / cls / name / "seqinfo.ini"
        info = configparser.ConfigParser()
        info.read(ini)
        seq = info["Sequence"]
        im_dir = (ini.parent / seq["imDir"]).absolute()  # not resolve(): DECISIONS.md
        out.append({"name": name, "length": int(seq["seqLength"]),
                    "width": int(seq["imWidth"]), "height": int(seq["imHeight"]),
                    "frames": [im_dir / f"{i:07d}{seq['imExt']}"
                               for i in range(1, int(seq["seqLength"]) + 1)]})
    return out


def track_sequence(frames: Iterable[np.ndarray], detector: Callable, tracker,
                   keep_classes: set[int]) -> np.ndarray:
    """(N, 8) rows frame, id, x1, y1, x2, y2, score, class over the whole sequence.
    Detections of classes outside `keep_classes` (not evaluated) never reach the tracker."""
    rows = []
    for f, img in enumerate(frames, 1):
        det = detector(img)
        det = det[np.isin(det[:, 5].astype(int), list(keep_classes))]
        for x1, y1, x2, y2, tid, score, cls in tracker.update(det):
            rows.append((f, tid, x1, y1, x2, y2, score, cls))
    return np.array(rows, dtype=float).reshape(-1, 8)


def read_frames(paths: list[Path]) -> Iterable[np.ndarray]:
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            raise FileNotFoundError(p)
        yield img


def write_tracks(rows: np.ndarray, seq: dict, ignore_file: Path, class_names: dict[int, str],
                 out_dir: Path) -> None:
    """<out_dir>/<class>/<tracker>/data/<seq>.txt in MOTChallenge format, ignored-region
    tracks dropped per frame with the dataset's rule (as for the ground truth)."""
    xywh = np.c_[rows[:, 2:4], rows[:, 4:6] - rows[:, 2:4]]
    keep = np.ones(len(rows), dtype=bool)
    ign = np.loadtxt(ignore_file, delimiter=",", ndmin=2) if ignore_file.stat().st_size else \
        np.zeros((0, 5))
    for frame in np.unique(ign[:, 0]).astype(int):
        mask = ignore_mask(ign[ign[:, 0] == frame, 1:5], seq["height"], seq["width"])
        idx = np.flatnonzero(rows[:, 0] == frame)
        keep[idx] = keep_outside(xywh[idx], mask)
    for cid, name in class_names.items():
        sel = keep & (rows[:, 7] == cid)
        path = out_dir / name / "data" / f"{seq['name']}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(
            f"{int(r[0])},{int(r[1])},{b[0]:.2f},{b[1]:.2f},{b[2]:.2f},{b[3]:.2f},{r[6]:.4f},-1,-1,-1\n"
            for r, b in zip(rows[sel], xywh[sel], strict=True)))


def timed_tracking(detector, make_tracker: Callable, frames: list[np.ndarray], warmup: int,
                   iters: int, keep_classes: set[int]) -> list[dict[str, float]]:
    """Per-frame stage times over consecutive frames, the tracker included. The tracker
    restarts whenever the frame list wraps, so it never sees a jump in time."""
    samples = []
    tracker = make_tracker()
    keep = list(keep_classes)
    for i in range(warmup + iters):
        if i % len(frames) == 0:
            tracker = make_tracker()
        det, t = detector.timed(frames[i % len(frames)])
        t0 = time.perf_counter()
        tracker.update(det[np.isin(det[:, 5].astype(int), keep)])
        t["tracking"] = (time.perf_counter() - t0) * 1e3
        if i >= warmup:
            samples.append(t)
    return samples
