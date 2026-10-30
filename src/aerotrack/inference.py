"""One detector pipeline for both accuracy and latency.

preprocess (letterbox, HWC->CHW, host->device) -> inference -> postprocess (NMS or the
end-to-end head's confidence filter, rescaling to the original image).

Accuracy and latency go through the SAME object, so a variant can never be measured fast
with one preprocessing and accurate with another. Backends: `torch` (cuda | mps | cpu) and
`onnxruntime` (CPU execution provider), and `tensorrt` (an engine built from the same
ONNX export, run through the same AutoBackend as torch; NVIDIA GPU only).

Input size is a fixed square letterbox (batch 1, static shape), which is what an exported
engine runs. Ultralytics' own `predict` pads to the minimum rectangle instead, so its mAP
can differ slightly from ours; ours is the deployment-faithful one.

Tiled inference (SAHI-style slicing): with `tiles=(rows, cols)` each frame is cut into a
fixed grid of overlapping crops, plus the whole frame if `tile_full`, each letterboxed to
`imgsz` and run as ONE batch (a static batch for ONNX). Per-tile detections are mapped back
to the frame and merged by class-wise greedy suppression on intersection over the smaller
box, which also removes the truncated boxes an object cut by a tile edge leaves behind.
A fixed grid, not fixed-size tiles: the cost per frame does not depend on its resolution.
"""

from __future__ import annotations

import hashlib
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from aerotrack.paths import WEIGHTS_DIR
from aerotrack.runtime import resolve_device, synchronize
from aerotrack.ultra import configure, load_yolo, resolve_weights

STAGES = ("preprocess", "inference", "postprocess")


def export_onnx(weights: str | Path, imgsz: int, max_det: int, batch: int = 1) -> Path:
    """Static-shape ONNX export, cached per (weights CONTENT, imgsz, max_det, batch).

    Keyed by a hash of the checkpoint, not its name: every fine-tune is called `best.pt`,
    and a name-keyed cache once timed a 3-epoch YOLO26n export in place of a 100-epoch
    YOLO26s (DECISIONS.md)."""
    weights = resolve_weights(weights)
    if not Path(weights).exists():
        load_yolo(weights)  # a bare release name: download it before hashing
    digest = hashlib.sha256(Path(weights).read_bytes()).hexdigest()[:12]
    suffix = f"_b{batch}" if batch != 1 else ""
    dest = WEIGHTS_DIR / "onnx" / f"{Path(weights).stem}_{digest}_{imgsz}_md{max_det}{suffix}.onnx"
    if dest.exists():
        return dest
    model = load_yolo(weights)
    out = Path(model.export(format="onnx", imgsz=imgsz, batch=batch, dynamic=False, simplify=True,
                            max_det=max_det, device="cpu", verbose=False))
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(out, dest)  # not rename: the cache may sit on another filesystem
    return dest


def merge_tiles(det: torch.Tensor, ios: float, max_det: int) -> torch.Tensor:
    """Greedy class-wise suppression of overlapping tile detections, highest score first.
    Overlap is intersection over the SMALLER box (SAHI's IOS): a box truncated by a tile
    edge lies mostly inside the full object's box, so its IoU with it can be low while
    its IOS is close to 1."""
    det = det[det[:, 4].argsort(descending=True)]
    b = det[:, :4]
    area = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    lt = torch.maximum(b[:, None, :2], b[None, :, :2])
    rb = torch.minimum(b[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    smaller = torch.minimum(area[:, None], area[None, :]).clamp(min=1e-9)
    over = (inter / smaller > ios) & (det[:, None, 5] == det[None, :, 5])
    keep = torch.ones(len(det), dtype=torch.bool)
    for i in range(len(det)):  # rows already sorted: i is kept iff no kept j < i covers it
        if keep[i]:
            keep[i + 1:] &= ~over[i, i + 1:]
    return det[keep][:max_det]


def export_engine(weights: str | Path, imgsz: int, max_det: int, batch: int, precision: str,
                  calibration_data: str | None = None) -> Path:
    """TensorRT engine (static batch), built by Ultralytics from its ONNX export, cached per
    (weights CONTENT, imgsz, max_det, batch, precision). An engine is tied to the GPU and
    TensorRT version it was built on, so the cache lives with the machine, never in git.
    INT8 calibrates on `calibration_data` (a dataset yaml; its val split, never test)."""
    if precision not in ("fp32", "fp16", "int8"):
        raise ValueError(f"precision must be fp32, fp16 or int8, not {precision!r}")
    if precision == "int8" and not calibration_data:
        raise ValueError("int8 needs calibration_data (a dataset yaml)")
    weights = resolve_weights(weights)
    if not Path(weights).exists():
        load_yolo(weights)
    digest = hashlib.sha256(Path(weights).read_bytes()).hexdigest()[:12]
    dest = (WEIGHTS_DIR / "engine" /
            f"{Path(weights).stem}_{digest}_{imgsz}_md{max_det}_b{batch}_{precision}.engine")
    if dest.exists():
        return dest
    model = load_yolo(weights)
    quantize = {"fp32": None, "fp16": 16, "int8": 8}[precision]
    out = Path(model.export(format="engine", imgsz=imgsz, batch=batch, dynamic=False,
                            simplify=True, max_det=max_det, device=0, quantize=quantize,
                            data=calibration_data if precision == "int8" else None,
                            verbose=False))
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(out, dest)  # not rename: the cache may sit on another filesystem
    return dest


class Detector:
    def __init__(
        self,
        weights: str | Path,
        backend: str = "torch",
        device: str = "auto",
        imgsz: int = 640,
        conf: float = 0.001,
        iou: float = 0.7,
        max_det: int = 300,
        half: bool = False,
        ort_threads: int | None = None,
        tiles: tuple[int, int] | None = None,
        tile_overlap: float = 0.2,
        tile_full: bool = True,
        merge_ios: float = 0.5,
        precision: str = "fp16",
        calibration_data: str | None = None,
    ) -> None:
        configure()
        from ultralytics.data.augment import LetterBox  # noqa: PLC0415

        self.weights = resolve_weights(weights)
        self.backend = backend
        self.imgsz, self.conf, self.iou, self.max_det = imgsz, conf, iou, max_det
        self.letterbox = LetterBox(new_shape=(imgsz, imgsz), auto=False, stride=32)
        self.tiles = tuple(tiles) if tiles else None
        self.tile_overlap, self.tile_full, self.merge_ios = tile_overlap, tile_full, merge_ios
        self.batch = (self.tiles[0] * self.tiles[1] + int(tile_full)) if self.tiles else 1
        self._crops: list[tuple[int, int, int, int]] = []  # set by preprocess, read by postprocess

        if backend in ("torch", "tensorrt"):
            from ultralytics.nn.autobackend import AutoBackend  # noqa: PLC0415

            self.device = resolve_device(device)
            path = self.weights
            if backend == "tensorrt":
                if not self.device.startswith("cuda"):
                    raise ValueError("the tensorrt backend needs an NVIDIA GPU")
                path = str(export_engine(self.weights, imgsz, max_det, self.batch, precision,
                                         calibration_data))
            # Same object for both: AutoBackend runs a .pt in torch and an .engine in TensorRT,
            # behind the same preprocessing and postprocessing.
            self.model = AutoBackend(path, device=torch.device(self.device), fp16=half,
                                     fuse=True, verbose=False)
            self.model.eval()
            self.half = bool(self.model.fp16)
            self.end2end = bool(getattr(self.model, "end2end", False))
            try:
                head = getattr(getattr(self.model, "model", None), "model", [None])[-1]
            except TypeError:  # an engine: max_det was fixed at export (export_engine)
                head = None
            if self.end2end and hasattr(head, "max_det"):
                head.max_det = max_det  # top-k inside the NMS-free head
            self.names = dict(self.model.names)
            self.runtime = {"backend": backend, "device": self.device, "half": self.half}
            if backend == "tensorrt":
                self.runtime.update({"precision": precision, "engine": path,
                                     "calibration_data": calibration_data
                                     if precision == "int8" else None})
        elif backend == "onnxruntime":
            import onnxruntime as ort  # noqa: PLC0415

            if device not in ("auto", "cpu"):
                raise ValueError("the onnxruntime backend is benchmarked on CPU only")
            self.device = "cpu"
            self.half = False
            onnx_path = export_onnx(self.weights, imgsz, max_det, self.batch)
            opts = ort.SessionOptions()
            if ort_threads:
                opts.intra_op_num_threads = ort_threads
            self.session = ort.InferenceSession(str(onnx_path), opts,
                                                providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            meta = self.session.get_modelmeta().custom_metadata_map
            self.names = {int(k): v for k, v in eval(meta["names"]).items()}  # noqa: S307
            out_shape = self.session.get_outputs()[0].shape
            self.end2end = len(out_shape) == 3 and out_shape[-1] == 6
            self.runtime = {
                "backend": "onnxruntime", "device": "cpu", "half": False,
                "onnx": str(onnx_path), "providers": self.session.get_providers(),
                "intra_op_num_threads": opts.intra_op_num_threads or "default",
            }
        else:
            raise ValueError(f"unknown backend {backend!r}")

        if self.tiles:
            self.runtime.update({"tiles": list(self.tiles), "tile_overlap": tile_overlap,
                                 "tile_full": tile_full, "merge_ios": merge_ios,
                                 "batch": self.batch})

    # -- stages ---------------------------------------------------------------------
    def crops(self, h: int, w: int) -> list[tuple[int, int, int, int]]:
        """(x0, y0, x1, y1) of each tile: a rows x cols grid whose neighbours overlap by
        `tile_overlap` of a tile, then the whole frame if `tile_full`."""
        if not self.tiles:
            return [(0, 0, w, h)]
        rows, cols = self.tiles
        ov = self.tile_overlap
        tw, th = w / (cols - (cols - 1) * ov), h / (rows - (rows - 1) * ov)
        out = [(round(c * tw * (1 - ov)), round(r * th * (1 - ov)),
                round(c * tw * (1 - ov) + tw), round(r * th * (1 - ov) + th))
               for r in range(rows) for c in range(cols)]
        out = [(x0, y0, min(x1, w), min(y1, h)) for x0, y0, x1, y1 in out]
        return out + [(0, 0, w, h)] if self.tile_full else out

    def preprocess(self, img_bgr: np.ndarray):
        self._crops = self.crops(*img_bgr.shape[:2])
        padded = np.stack([self.letterbox(image=img_bgr[y0:y1, x0:x1])
                           for x0, y0, x1, y1 in self._crops])
        chw = np.ascontiguousarray(padded[..., ::-1].transpose(0, 3, 1, 2))
        if self.backend == "torch":
            t = torch.from_numpy(chw).to(self.device, non_blocking=True)
            return (t.half() if self.half else t.float()).div_(255)
        return chw.astype(np.float32) / 255.0

    def infer(self, inp):
        if self.backend == "torch":
            with torch.inference_mode():
                return self.model(inp)
        return torch.from_numpy(self.session.run(None, {self.input_name: inp})[0])

    def postprocess(self, raw, orig_shape: tuple[int, int]) -> np.ndarray:
        from ultralytics.utils import nms, ops  # noqa: PLC0415

        if self.device == "mps":
            # Measured on an M3 Pro: NMS / top-k filtering on MPS costs 12-18 ms per image
            # (kernel-launch overhead of many small ops), against ~1.2 ms for copying the
            # raw output to the host and filtering there. The copy stays inside this stage.
            raw = raw[0] if isinstance(raw, list | tuple) else raw
            raw = raw.float().cpu()
        dets = nms.non_max_suppression(raw, self.conf, self.iou, max_det=self.max_det,
                                       end2end=self.end2end)
        for det, (x0, y0, x1, y1) in zip(dets, self._crops, strict=True):
            det[:, :4] = ops.scale_boxes((self.imgsz, self.imgsz), det[:, :4], (y1 - y0, x1 - x0))
            det[:, [0, 2]] += x0
            det[:, [1, 3]] += y0
        det = dets[0] if len(dets) == 1 else merge_tiles(torch.cat(dets), self.merge_ios,
                                                          self.max_det)
        return det.cpu().numpy()  # (N, 6): x1, y1, x2, y2, conf, class

    # -- public ---------------------------------------------------------------------
    def __call__(self, img_bgr: np.ndarray) -> np.ndarray:
        return self.postprocess(self.infer(self.preprocess(img_bgr)), img_bgr.shape[:2])

    def timed(self, img_bgr: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        """Same as __call__, with per-stage wall time in ms. Each stage ends with a device
        sync, otherwise asynchronous GPU work would be billed to the next stage."""
        t0 = time.perf_counter()
        inp = self.preprocess(img_bgr)
        synchronize(self.device)
        t1 = time.perf_counter()
        raw = self.infer(inp)
        synchronize(self.device)
        t2 = time.perf_counter()
        det = self.postprocess(raw, img_bgr.shape[:2])  # ends with .cpu(): implicit sync
        t3 = time.perf_counter()
        return det, {"preprocess": (t1 - t0) * 1e3, "inference": (t2 - t1) * 1e3,
                     "postprocess": (t3 - t2) * 1e3}
