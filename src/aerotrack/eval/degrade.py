"""Synthetic image degradations for the robustness sweep (week 11).

Applied to the decoded BGR frame before the detector, as a camera or a link would degrade
it. Each kind has three levels of a physical parameter, not an abstract "severity":

  motion_blur   linear blur of `length` px at a random angle (platform vibration, fast pan)
  noise         additive Gaussian noise of std `sigma` on the 0-255 scale (sensor, gain)
  jpeg          re-encoding at JPEG `quality` (compressed video link)
  low_light     exposure scaled by `gain`, then sensor noise (sigma 3) and 8-bit rounding

Random parts (blur angle, noise) are drawn from a generator seeded by the frame's own
content, so every model sees exactly the same degraded frame and reruns are identical.
"""

from __future__ import annotations

import zlib

import cv2
import numpy as np

KINDS = {
    "motion_blur": "length",
    "noise": "sigma",
    "jpeg": "quality",
    "low_light": "gain",
}


def _rng(img: np.ndarray) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32(np.ascontiguousarray(img[::31, ::31]).tobytes()))


def motion_blur(img: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0 / length
    rot = cv2.getRotationMatrix2D(((length - 1) / 2, (length - 1) / 2), rng.uniform(0, 180), 1.0)
    k = cv2.warpAffine(k, rot, (length, length))
    return cv2.filter2D(img, -1, k / k.sum(), borderType=cv2.BORDER_REFLECT)


def _noisy(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    out = img.astype(np.float32) + rng.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def degrade(img: np.ndarray, kind: str, level: float) -> np.ndarray:
    rng = _rng(img)
    if kind == "motion_blur":
        return motion_blur(img, int(level), rng)
    if kind == "noise":
        return _noisy(img, level, rng)
    if kind == "jpeg":
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(level)])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if kind == "low_light":
        return _noisy(img.astype(np.float32) * level, 3.0, rng)
    raise ValueError(f"unknown degradation {kind!r}, expected one of {sorted(KINDS)}")


def label(kind: str, level: float) -> str:
    return f"{kind} {KINDS[kind]}={level:g}"
