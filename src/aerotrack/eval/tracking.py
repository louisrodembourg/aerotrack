"""TrackEval wrapper for the per-class MOTChallenge benchmarks written by aerotrack.data.mot.

Week 1 only needs the self-check (ground truth scored as a tracker); the tracker
evaluation of week 6 goes through `run_trackeval` with the same configuration.
"""

from __future__ import annotations

import contextlib
import io
import shutil
import tempfile
from pathlib import Path

import numpy as np

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
