"""Export for the README demo video (demo/, rendered with Remotion).

The video draws nothing it was not given: the tracks and the per-frame stage times come
from the reference pipeline (`TrackingPipeline` around `inference.Detector`) run here, in a
tracked run, on clips of sequences the detector never saw; the headline numbers are read
from reports/*/results.json with their run ids and devices. Remotion only animates them.

Output (`out_dir`, git-ignored: the frames are VisDrone images, non-commercial licence):
clips/<sequence>/<i>.jpg, the shown frames, and demo.json, everything the video displays.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from aerotrack.eval.harness import check_power
from aerotrack.eval.report import _device, _versions
from aerotrack.eval.track_harness import make_tracker_factory
from aerotrack.eval.tracking import (
    MOT_ROOTS,
    benchmark_classes,
    read_frames,
    seen_by_detector,
    sequences,
)
from aerotrack.experiments import find_train_run, start_run
from aerotrack.inference import STAGES, Detector
from aerotrack.paths import REPO_ROOT
from aerotrack.runtime import environment_info, resolve_device, seed_everything
from aerotrack.track.pipeline import TrackingPipeline


def smallest_steady_track(tracks: list[np.ndarray], presence: float = 0.9) -> dict | None:
    """The track with the smallest median box among those present in at least `presence`
    of the frames: a small object followed through the clip, for the zoom."""
    boxes: dict[int, list[tuple[float, float]]] = {}
    for frame in tracks:
        for x1, y1, x2, y2, tid, _, _ in frame:
            boxes.setdefault(int(tid), []).append((x2 - x1, y2 - y1))
    steady = {t: np.median(np.array(b), axis=0) for t, b in boxes.items()
              if len(b) >= presence * len(tracks)}
    if not steady:
        return None
    tid = min(steady, key=lambda t: steady[t][0] * steady[t][1])
    w, h = steady[tid]
    return {"id": tid, "w": round(float(w), 1), "h": round(float(h), 1)}


def headline(reports: dict) -> dict:
    """Numbers of the closing slide, each with the run it comes from."""
    zs = json.loads((REPO_ROOT / reports["zero_shot"]).read_text())
    by_model = {r["model"]: r for r in zs["results"]}
    zero_shot = [{"model": m, "mAP50-95": by_model[m]["accuracy"]["mAP50-95"],
                  "run_id": by_model[m]["run_id"]}
                 for m in ("yolo26n-coco", "visdrone-yolo26n-640")]
    trk = json.loads((REPO_ROOT / reports["tracking"]).read_text())
    tracking = []
    for r in trk["results"]:
        lat = r["latency"][0]
        basis = lat.get("budget_basis", "median")
        tracking.append({"model": r["model"], "imgsz": r["accuracy"]["imgsz"],
                         "HOTA": r["accuracy"]["HOTA"], "IDF1": r["accuracy"]["IDF1"],
                         "fps": lat[f"fps_{basis}"], "total_ms": lat["total"][f"{basis}_ms"],
                         "backend": lat["runtime"]["backend"], "device": _device(lat["env"]),
                         "versions": _versions(lat), "run_id": r["run_id"]})
    return {"zero_shot": {"report": reports["zero_shot"], "split": "VisDrone-DET test-dev",
                          "models": zero_shot},
            "tracking": {"report": reports["tracking"], "split": "VisDrone-MOT test-dev",
                         "budget_fps": trk["budget_fps"], "models": tracking}}


def export_demo(cfg: dict) -> Path:
    check_power(cfg.get("latency", {}))  # the video shows the per-frame times
    seed_everything(cfg.get("seed", 0))
    ds = cfg["dataset"]
    split_dir = MOT_ROOTS[ds["mot"]] / ds["split"]
    by_name = {s["name"]: s for s in sequences(split_dir)}
    for clip in cfg["clips"]:
        if clip["sequence"] not in by_name:
            raise SystemExit(f"{clip['sequence']} is not in {split_dir}")
        if seen_by_detector(clip["sequence"]):
            raise SystemExit(f"{clip['sequence']}: the detector saw this video (DET train/val)")

    device = resolve_device(cfg.get("device", "auto"))
    model, det = cfg["model"], cfg["detection"]
    train = find_train_run(model["run"])
    detector = Detector(train["summary"]["weights"], device=device, imgsz=model["imgsz"],
                        conf=det["conf"], iou=det["iou"], max_det=det["max_det"])
    classes = benchmark_classes(split_dir)
    keep = {i for i, n in detector.names.items() if n in classes}
    make_tracker = make_tracker_factory(cfg["tracker"])
    out_dir = REPO_ROOT / cfg["out_dir"]

    with start_run(cfg["name"], "demo-export", cfg, seed=cfg.get("seed"), device=device,
                   tags=(cfg["name"],)) as run:
        shutil.rmtree(out_dir / "clips", ignore_errors=True)
        clips = []
        for clip in cfg["clips"]:
            seq = by_name[clip["sequence"]]
            first, n = clip["start"] - 1, clip["frames"]
            if first + n > seq["length"]:
                raise SystemExit(f"{seq['name']}: {seq['length']} frames, clip ends at {first + n}")
            pipe = TrackingPipeline(detector, make_tracker, keep)
            for img in read_frames(seq["frames"][: cfg["warmup"]]):
                pipe.step(img)
            pipe.reset()  # tracking restarts at frame 1: the warm-up only warms the device

            clip_dir = out_dir / "clips" / seq["name"]
            clip_dir.mkdir(parents=True)
            tracks, timing = [], []
            for i, img in enumerate(read_frames(seq["frames"][: first + n])):
                t, ms = pipe.timed_step(img)
                if i < first:
                    continue
                cv2.imwrite(str(clip_dir / f"{i - first:04d}.jpg"), img,
                            [cv2.IMWRITE_JPEG_QUALITY, cfg["jpeg_quality"]])
                tracks.append(t)
                timing.append([round(ms[s], 2) for s in (*STAGES, "tracking")])
            print(f"[demo] {seq['name']}: {n} frames, "
                  f"median {np.median([sum(t) for t in timing]):.1f} ms", flush=True)
            clips.append({
                "sequence": seq["name"], "caption": clip["caption"], "dir": f"clips/{seq['name']}",
                "width": seq["width"], "height": seq["height"], "frames": n,
                "first_frame": clip["start"],
                "zoom": smallest_steady_track(tracks) if clip.get("zoom") else None,
                "stages": [*STAGES, "tracking"], "timing_ms": timing,
                # x1, y1, x2, y2, id, class id (scores are not shown)
                "tracks": [[[round(float(v), 1) for v in r[:4]] + [int(r[4]), int(r[6])]
                            for r in t] for t in tracks]})

        totals = [sum(t) for c in clips for t in c["timing_ms"]]
        env = environment_info(device)
        payload = {
            "run_id": run.id, "fps": cfg["fps"], "model": model["label"],
            "train_run": train["id"], "split": f"VisDrone-MOT {ds['split']}",
            "classes": {i: detector.names[i] for i in sorted(keep)},
            "device": _device(env),
            "versions": _versions({"env": env, "runtime": detector.runtime}),
            "median_ms": float(np.median(totals)), "p95_ms": float(np.percentile(totals, 95)),
            "clips": clips, "headline": headline(cfg["reports"])}
        out = out_dir / "demo.json"
        out.write_text(json.dumps(payload, separators=(",", ":")))
        run.log({"median_ms": payload["median_ms"], "p95_ms": payload["p95_ms"]})
        run.summary({k: v for k, v in payload.items() if k not in ("clips", "headline")})
        shutil.copy(out, run.dir / "demo.json")
        run.artifact(run.dir / "demo.json", kind="demo")
    print(f"demo export: {out}")
    return out
