"""Evaluation harness: accuracy + latency for each model of a config, one tracked run per
model, one JSON + Markdown report per config (reports/<name>/)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import yaml

from aerotrack.data.classes import get_class_map
from aerotrack.data.gt import SplitGT
from aerotrack.eval.detection import evaluate_detection
from aerotrack.eval.latency import benchmark
from aerotrack.eval.report import render_markdown
from aerotrack.experiments import find_train_run, start_run
from aerotrack.inference import Detector
from aerotrack.paths import CONFIGS_DIR, PROCESSED_DIR, REPO_ROOT, from_data_relative
from aerotrack.runtime import environment_info, resolve_device, seed_everything


def project_config() -> dict:
    return yaml.safe_load((CONFIGS_DIR / "project.yaml").read_text())


def _progress(label: str):
    start = time.perf_counter()

    def report(i: int, n: int) -> None:
        if i == n or i % max(1, n // 10) == 0:
            print(f"  {label}: {i}/{n} images ({time.perf_counter() - start:.0f}s)", flush=True)

    return report


def evaluate_model(model: dict, cfg: dict, gt: SplitGT, budget_fps: float) -> dict:
    lat_cfg = cfg.get("latency", {})
    # A model may override detection keys, e.g. a resolution sweep scores each model at
    # the input size it was trained at; the value used is recorded in its accuracy row.
    det_cfg = {**cfg["detection"], **{k: model[k] for k in cfg["detection"] if k in model}}
    device = resolve_device(cfg.get("device", "auto"))
    if "run" in model:
        # A fine-tuned model is named by the training run that produced it, so the report
        # links accuracy and latency back to that run's config and commit.
        train = find_train_run(model["run"])
        model = {**model, "weights": train["summary"]["weights"], "train_run": train["id"]}
        name = model.get("name", train["name"])
    else:
        name = model.get("name", Path(model["weights"]).stem)
    weights = model["weights"]
    cmap = get_class_map(cfg["dataset"]["class_map"])

    with start_run(f"{cfg['name']}-{name}", "evaluate", {**cfg, "model": model},
                   seed=cfg.get("seed"), device=device, tags=(cfg["name"],)) as run:
        result = {"model": name, "weights": str(weights), "run_id": run.id,
                  "train_run": model.get("train_run"),
                  "wandb_url": run.record.get("wandb", {}).get("url"),
                  "dataset": {**cfg["dataset"], "n_images": len(gt.images)}}

        detector = Detector(weights, backend="torch", device=device, imgsz=det_cfg["imgsz"],
                            conf=det_cfg["conf"], iou=det_cfg["iou"], max_det=det_cfg["max_det"],
                            half=det_cfg.get("half", False))
        print(f"[{name}] accuracy on {len(gt.images)} images, device={device}", flush=True)
        acc = evaluate_detection(gt, cmap, detector, detector.names, det_cfg["max_det"],
                                 _progress(name))
        acc.update({"imgsz": det_cfg["imgsz"], "conf": det_cfg["conf"], "iou": det_cfg["iou"],
                    "runtime": detector.runtime})
        result["accuracy"] = acc
        run.log({k: v for k, v in acc.items() if isinstance(v, float)})

        result["latency"] = []
        if lat_cfg.get("enabled", True):
            images = [cv2.imread(str(from_data_relative(im.file)))
                      for im in gt.subset(lat_cfg["n_images"], seed=cfg.get("seed", 0)).images]
            for backend in lat_cfg["backends"]:
                bdev = device if backend == "torch" else "cpu"
                d = Detector(weights, backend=backend, device=bdev, imgsz=det_cfg["imgsz"],
                             conf=lat_cfg["conf"], iou=det_cfg["iou"], max_det=det_cfg["max_det"],
                             half=det_cfg.get("half", False) and backend == "torch",
                             ort_threads=lat_cfg.get("ort_threads"))
                print(f"[{name}] latency backend={backend} device={bdev}", flush=True)
                lat = benchmark(d, images, lat_cfg["warmup"], lat_cfg["iters"])
                lat["env"] = environment_info(bdev)
                lat["stages_measured"] = ["preprocess", "inference", "postprocess"]
                lat["tracking_included"] = False
                lat["budget_fps"] = budget_fps
                lat["meets_budget_median"] = lat["fps_median"] >= budget_fps
                result["latency"].append(lat)
                run.log({f"latency/{backend}/fps_median": lat["fps_median"],
                         f"latency/{backend}/total_p95_ms": lat["total"]["p95_ms"]})
                del d

        run.summary(result)
        out = run.dir / "result.json"
        out.write_text(json.dumps(result, indent=2))
        run.artifact(out, kind="evaluation")
        return result


def check_power(lat_cfg: dict) -> None:
    """Refuse to time a throttled Mac. On battery with Low Power Mode, YOLO26n on MPS
    measured 20.5 ms against 7.7 ms plugged in (DECISIONS.md): such a number would be
    published as the device's latency. Checked before the accuracy pass, which is long."""
    if not lat_cfg.get("enabled", True) or lat_cfg.get("allow_throttled", False):
        return
    power = environment_info().get("macos_power")
    if power and (power["source"] != "ac" or power["low_power_mode"]):
        raise SystemExit(f"latency on a throttled Mac ({power}): plug in the charger and turn "
                         "Low Power Mode off, or set latency.allow_throttled=true to measure "
                         "that state on purpose (it is recorded in every latency row)")


def run_evaluation(cfg: dict) -> Path:
    check_power(cfg.get("latency", {}))
    seed_everything(cfg.get("seed", 0))
    ds = cfg["dataset"]
    gt = SplitGT.load(PROCESSED_DIR / ds["gt"]).subset(ds.get("limit"), seed=cfg.get("seed", 0))
    budget = project_config()["budget_fps"]
    results = [evaluate_model(m, cfg, gt, budget) for m in cfg["models"]]

    out_dir = REPO_ROOT / cfg.get("report_dir", f"reports/{cfg['name']}")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"name": cfg["name"], "config": cfg, "budget_fps": budget, "results": results}
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2, default=str))
    (out_dir / "results.md").write_text(render_markdown(payload))
    print(f"report: {out_dir / 'results.md'}")
    return out_dir
