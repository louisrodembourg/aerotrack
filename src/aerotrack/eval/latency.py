"""Batch-1 latency benchmark, decomposed per pipeline stage.

Images are decoded into memory beforehand: disk and JPEG decoding are not part of the
pipeline budget (a camera delivers raw frames). Real validation images are cycled rather
than one repeated frame, because postprocessing cost grows with the number of candidates.
"""

from __future__ import annotations

import resource
import sys

import numpy as np
import torch

from aerotrack.inference import STAGES, Detector


def summarize(samples: list[dict[str, float]], stages: tuple[str, ...] = STAGES) -> dict:
    """Per-stage and end-to-end statistics. The end-to-end figure is computed per
    iteration and then summarised: summing per-stage medians would understate p95."""
    out = {}
    columns = {s: np.array([x[s] for x in samples]) for s in stages}
    columns["total"] = sum(columns[s] for s in stages)
    for name, v in columns.items():
        out[name] = {
            "median_ms": float(np.median(v)),
            "p95_ms": float(np.percentile(v, 95)),
            "mean_ms": float(v.mean()),
            "std_ms": float(v.std()),
            "min_ms": float(v.min()),
            "max_ms": float(v.max()),
        }
    out["fps_median"] = 1000.0 / out["total"]["median_ms"]
    out["fps_p95"] = 1000.0 / out["total"]["p95_ms"]
    out["iterations"] = len(samples)
    return out


def peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10  # bytes vs KiB


def device_memory_mb(device: str) -> float | None:
    if device.startswith("cuda"):
        return torch.cuda.max_memory_allocated() / 2**20
    if device == "mps":
        return torch.mps.driver_allocated_memory() / 2**20
    return None


def benchmark(detector: Detector, images: list[np.ndarray], warmup: int, iters: int) -> dict:
    if not images:
        raise ValueError("no images to benchmark on")
    for i in range(warmup):
        detector.timed(images[i % len(images)])
    if detector.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    samples = [detector.timed(images[i % len(images)])[1] for i in range(iters)]
    result = summarize(samples)
    result.update({
        "warmup": warmup,
        "batch": 1,
        "imgsz": detector.imgsz,
        "conf": detector.conf,
        "n_distinct_images": len(images),
        "peak_process_rss_mb": peak_rss_mb(),
        "device_memory_mb": device_memory_mb(detector.device),
        "runtime": detector.runtime,
        "end2end_head": detector.end2end,
    })
    return result
