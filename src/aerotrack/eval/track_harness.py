"""Tracking evaluation harness: for each model of a config, detector -> tracker over every
sequence of a MOT split, scored per class with TrackEval (HOTA, MOTA, IDF1, IDSW), then
end-to-end latency with the tracker inside the timed loop. One tracked run per model, one
JSON + Markdown report per config (reports/<name>/), like the detection harness."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

from aerotrack.eval.harness import TILING, check_power, project_config
from aerotrack.eval.latency import device_memory_mb, peak_rss_mb, summarize
from aerotrack.eval.report import render_tracking_markdown
from aerotrack.eval.tracking import (
    MOT_ROOTS,
    benchmark_classes,
    read_frames,
    run_trackeval,
    sequences,
    timed_tracking,
    track_sequence,
    write_tracks,
)
from aerotrack.experiments import find_train_run, start_run
from aerotrack.inference import STAGES, Detector
from aerotrack.paths import REPO_ROOT
from aerotrack.runtime import environment_info, resolve_device, seed_everything
from aerotrack.track.bytetrack import ByteTrack

TRACKERS = {"bytetrack": ByteTrack}


def make_tracker_factory(cfg: dict):
    params = {k: v for k, v in cfg.items() if k != "type"}
    return lambda: TRACKERS[cfg["type"]](**params)


def evaluate_tracking_model(model: dict, cfg: dict, split_dir: Path, budget: float) -> dict:
    det_cfg = {**cfg["detection"],
               **{k: model[k] for k in (*cfg["detection"], *TILING) if k in model}}
    tiling = {k: det_cfg[k] for k in TILING if k in det_cfg}
    trk_cfg = {**cfg["tracker"], **model.get("tracker", {})}
    make_tracker = make_tracker_factory(trk_cfg)
    device = resolve_device(cfg.get("device", "auto"))
    train = find_train_run(model["run"])
    weights = train["summary"]["weights"]
    name = model.get("name", train["name"])
    classes = benchmark_classes(split_dir)
    lat_cfg = cfg.get("latency", {})

    with start_run(f"{cfg['name']}-{name}", "evaluate-tracking", {**cfg, "model": model},
                   seed=cfg.get("seed"), device=device, tags=(cfg["name"],)) as run:
        # The detector keeps everything down to the tracker's low threshold: ByteTrack's
        # second association needs the weak detections.
        detector = Detector(weights, backend="torch", device=device, imgsz=det_cfg["imgsz"],
                            conf=det_cfg["conf"], iou=det_cfg["iou"], max_det=det_cfg["max_det"],
                            **tiling)
        keep = {i for i, n in detector.names.items() if n in classes}
        missing = set(classes) - {detector.names[i] for i in keep}
        if missing:
            raise SystemExit(f"{name}: no model class for evaluated class(es) {sorted(missing)}")
        class_ids = {i: detector.names[i] for i in sorted(keep)}

        tmp = Path(tempfile.mkdtemp(prefix="aerotrack-mot-"))
        try:
            seqs = sequences(split_dir)
            for k, seq in enumerate(seqs, 1):
                print(f"[{name}] {seq['name']} ({k}/{len(seqs)}, {seq['length']} frames)",
                      flush=True)
                rows = track_sequence(read_frames(seq["frames"]), detector, make_tracker(), keep)
                write_tracks(rows, seq, split_dir / "ignore" / f"{seq['name']}.txt", class_ids,
                             tmp)
            per_class = {}
            for cls in classes:
                per_class[cls] = run_trackeval(split_dir, cls, tmp, cls)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        mean = {m: float(np.mean([v[m] for v in per_class.values()]))
                for m in ("HOTA", "MOTA", "IDF1")}
        mean["IDSW"] = int(sum(v["IDSW"] for v in per_class.values()))
        acc = {**mean, "per_class": per_class, "classes": classes,
               "split": str(split_dir.relative_to(REPO_ROOT)) if split_dir.is_relative_to(REPO_ROOT)
               else str(split_dir), "imgsz": det_cfg["imgsz"], "conf": det_cfg["conf"],
               "tracker": trk_cfg, "runtime": detector.runtime}
        run.log({k: v for k, v in mean.items()})
        result = {"model": name, "weights": str(weights), "run_id": run.id,
                  "train_run": train["id"], "accuracy": acc, "latency": []}

        if lat_cfg.get("enabled", True):
            seqs = sequences(split_dir)
            # The first sequence unless one is named: the frames only need to be consecutive.
            seq = next((s for s in seqs if s["name"] == lat_cfg.get("sequence")), seqs[0])
            frames = [cv2.imread(str(p)) for p in seq["frames"][: lat_cfg["n_frames"]]]
            for backend in lat_cfg["backends"]:
                bdev = device if backend == "torch" else "cpu"
                d = Detector(weights, backend=backend, device=bdev, imgsz=det_cfg["imgsz"],
                             conf=det_cfg["conf"], iou=det_cfg["iou"], max_det=det_cfg["max_det"],
                             ort_threads=lat_cfg.get("ort_threads"), **tiling)
                print(f"[{name}] end-to-end latency backend={backend} device={bdev}", flush=True)
                samples = timed_tracking(d, make_tracker, frames, lat_cfg["warmup"],
                                         lat_cfg["iters"], keep)
                lat = summarize(samples, (*STAGES, "tracking"))
                lat.update({"warmup": lat_cfg["warmup"], "batch": 1, "imgsz": d.imgsz,
                            "conf": d.conf, "sequence": seq["name"],
                            "n_distinct_frames": len(frames), "runtime": d.runtime,
                            "peak_process_rss_mb": peak_rss_mb(),
                            "device_memory_mb": device_memory_mb(d.device),
                            "env": environment_info(bdev),
                            "stages_measured": [*STAGES, "tracking"],
                            "tracking_included": True, "budget_fps": budget,
                            "meets_budget_median": lat["fps_median"] >= budget})
                result["latency"].append(lat)
                run.log({f"latency/{backend}/fps_median": lat["fps_median"],
                         f"latency/{backend}/total_p95_ms": lat["total"]["p95_ms"]})
                del d

        run.summary(result)
        out = run.dir / "result.json"
        out.write_text(json.dumps(result, indent=2, default=str))
        run.artifact(out, kind="evaluation")
        return result


def run_tracking_evaluation(cfg: dict) -> Path:
    check_power(cfg.get("latency", {}))
    seed_everything(cfg.get("seed", 0))
    ds = cfg["dataset"]
    split_dir = MOT_ROOTS[ds["mot"]] / ds["split"]
    budget = project_config()["budget_fps"]
    results = [evaluate_tracking_model(m, cfg, split_dir, budget) for m in cfg["models"]]

    out_dir = REPO_ROOT / cfg.get("report_dir", f"reports/{cfg['name']}")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"name": cfg["name"], "kind": "tracking", "config": cfg, "budget_fps": budget,
               "results": results}
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2, default=str))
    (out_dir / "results.md").write_text(render_tracking_markdown(payload))
    print(f"report: {out_dir / 'results.md'}")
    return out_dir
