"""Tracker threshold search on MOT sequences the detector never saw.

The detector runs once per model over the tuning sequences and its detections are kept in
memory (down to the lowest threshold of the grid); every grid point then only replays
the tracker and TrackEval, which is what makes a grid affordable. Sequences come from
several splits (MOT train and val), gathered into one temporary benchmark of symlinks so
TrackEval scores them as one set. Selection is on mean HOTA over classes; test-dev is
never touched here.
"""

from __future__ import annotations

import itertools
import json
import shutil
import tempfile
from functools import partial
from pathlib import Path

import numpy as np

from aerotrack.eval.harness import TILING
from aerotrack.eval.report import _pct
from aerotrack.eval.track_harness import TRACKERS
from aerotrack.eval.tracking import (
    MOT_ROOTS,
    benchmark_classes,
    read_frames,
    run_trackeval,
    seen_by_detector,
    sequences,
    track_sequence,
    write_tracks,
)
from aerotrack.experiments import find_train_run, start_run
from aerotrack.inference import Detector
from aerotrack.paths import REPO_ROOT
from aerotrack.runtime import resolve_device, seed_everything
from aerotrack.track.pipeline import TrackingPipeline


def tuning_sequences(mot: str, splits: list[str]) -> list[tuple[Path, dict]]:
    out = []
    for split in splits:
        split_dir = MOT_ROOTS[mot] / split
        out += [(split_dir, s) for s in sequences(split_dir) if not seen_by_detector(s["name"])]
    return out


def gather_benchmark(seqs: list[tuple[Path, dict]], classes: list[str], root: Path) -> Path:
    """One benchmark over sequences of several splits: symlinked GT dirs + one seqmap."""
    for split_dir, s in seqs:
        for cls in classes:
            (root / cls).mkdir(parents=True, exist_ok=True)
            (root / cls / s["name"]).symlink_to((split_dir / cls / s["name"]).absolute())
    (root / "seqmap.txt").write_text("name\n" + "\n".join(s["name"] for _, s in seqs) + "\n")
    return root


def grid(space: dict) -> list[dict]:
    """Cartesian product; `new_offset` sets new = high + offset (a new track needs at
    least a first-stage score)."""
    keys = list(space)
    points = []
    for values in itertools.product(*(space[k] for k in keys)):
        p = dict(zip(keys, values, strict=True))
        p["new"] = round(p["high"] + p.pop("new_offset", 0.0), 3)
        points.append(p)
    return points


def tune_model(model: dict, cfg: dict) -> dict:
    det_cfg = {**cfg["detection"], **{k: model[k] for k in (*cfg["detection"], *TILING) if k in model}}
    device = resolve_device(cfg.get("device", "auto"))
    train = find_train_run(model["run"])
    name = model.get("name", train["name"])
    seqs = tuning_sequences(cfg["dataset"]["mot"], cfg["dataset"]["splits"])
    classes = benchmark_classes(seqs[0][0])
    points = grid(cfg["grid"])
    if min(p["low"] for p in points) < det_cfg["conf"]:
        raise SystemExit("detection.conf must not exceed the grid's lowest `low`")

    with start_run(f"{cfg['name']}-{name}", "tune-tracker", {**cfg, "model": model},
                   seed=cfg.get("seed"), device=device, tags=(cfg["name"],)) as run:
        det = Detector(train["summary"]["weights"], backend="torch", device=device,
                       imgsz=det_cfg["imgsz"], conf=det_cfg["conf"], iou=det_cfg["iou"],
                       max_det=det_cfg["max_det"])
        keep = [i for i, n in det.names.items() if n in classes]
        class_ids = {i: det.names[i] for i in sorted(keep)}
        cache = {}
        for _, s in seqs:
            print(f"[{name}] detecting {s['name']} ({s['length']} frames)", flush=True)
            cache[s["name"]] = [d[np.isin(d[:, 5].astype(int), keep)]
                                for d in map(det, read_frames(s["frames"]))]

        tmp = Path(tempfile.mkdtemp(prefix="aerotrack-tune-"))
        trials = []
        try:
            bench = gather_benchmark(seqs, classes, tmp / "gt")
            for k, p in enumerate(points, 1):
                tracker_cfg = {**cfg["tracker"], **p}
                params = {kk: v for kk, v in tracker_cfg.items() if kk != "type"}
                out = tmp / f"trial{k}"
                for split_dir, s in seqs:
                    # Cached detections replayed through the same loop as the evaluation.
                    make = partial(TRACKERS[tracker_cfg["type"]], **params)
                    pipe = TrackingPipeline(lambda d: d, make, set(keep))
                    rows = track_sequence(cache[s["name"]], pipe)
                    write_tracks(rows, s, split_dir / "ignore" / f"{s['name']}.txt",
                                 class_ids, out)
                per_class = {c: run_trackeval(bench, c, out, c) for c in classes}
                score = {m: float(np.mean([v[m] for v in per_class.values()]))
                         for m in ("HOTA", "MOTA", "IDF1")}
                score["IDSW"] = int(sum(v["IDSW"] for v in per_class.values()))
                trials.append({"tracker": tracker_cfg, **score})
                run.log({f"trial/{m}": v for m, v in score.items()}, step=k)
                print(f"[{name}] {k}/{len(points)} {p} HOTA={score['HOTA']:.4f} "
                      f"MOTA={score['MOTA']:.4f}", flush=True)
                shutil.rmtree(out, ignore_errors=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        best = max(trials, key=lambda t: t["HOTA"])
        result = {"model": name, "run_id": run.id, "train_run": train["id"],
                  "imgsz": det_cfg["imgsz"], "conf": det_cfg["conf"],
                  "sequences": [s["name"] for _, s in seqs],
                  "frames": sum(s["length"] for _, s in seqs),
                  "best": best, "trials": sorted(trials, key=lambda t: -t["HOTA"])}
        run.summary({"best": best})
        out = run.dir / "result.json"
        out.write_text(json.dumps(result, indent=2))
        run.artifact(out, kind="evaluation")
        return result


def render(payload: dict) -> str:
    keys = [k for k in payload["config"]["grid"] if k != "new_offset"] + ["new"]
    blocks = [f"""# {payload['name']}

Generated by `aerotrack tune-tracker`; one tracked run per model. Grid search of tracker
thresholds on VisDrone-MOT sequences whose source video has no frame in VisDrone-DET
train or val; selection on mean HOTA over classes. Test-dev is not used here.
"""]
    for r in payload["results"]:
        rows = ["| " + " | ".join(keys) + " | HOTA | MOTA | IDF1 | IDSW |",
                "|" + "---|" * (len(keys) + 4)]
        for t in r["trials"]:
            rows.append("| " + " | ".join(str(t["tracker"][k]) for k in keys) +
                        f" | {_pct(t['HOTA'])} | {_pct(t['MOTA'])} | {_pct(t['IDF1'])} | {t['IDSW']} |")
        blocks.append(f"""## {r['model']} (imgsz {r['imgsz']}, run `{r['run_id']}`)

{len(r['sequences'])} sequences, {r['frames']} frames: {', '.join(r['sequences'])}.
Best: {', '.join(f"{k}={r['best']['tracker'][k]}" for k in keys)}.

{chr(10).join(rows)}
""")
    return "\n".join(blocks)


def run_tuning(cfg: dict) -> Path:
    seed_everything(cfg.get("seed", 0))
    results = [tune_model(m, cfg) for m in cfg["models"]]
    out_dir = REPO_ROOT / cfg.get("report_dir", f"reports/{cfg['name']}")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"name": cfg["name"], "kind": "tuning", "config": cfg, "results": results}
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2, default=str))
    (out_dir / "results.md").write_text(render(payload))
    print(f"report: {out_dir / 'results.md'}")
    return out_dir
