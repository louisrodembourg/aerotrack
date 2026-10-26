"""One detector pipeline for both accuracy and latency.

preprocess (letterbox, HWC->CHW, host->device) -> inference -> postprocess (NMS or the
end-to-end head's confidence filter, rescaling to the original image).

Accuracy and latency go through the SAME object, so a variant can never be measured fast
with one preprocessing and accurate with another. Backends: `torch` (cuda | mps | cpu) and
`onnxruntime` (CPU execution provider). TensorRT is added in week 8 behind the same API.

Input size is a fixed square letterbox (batch 1, static shape), which is what an exported
engine runs. Ultralytics' own `predict` pads to the minimum rectangle instead, so its mAP
can differ slightly from ours; ours is the deployment-faithful one.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import torch

from aerotrack.paths import WEIGHTS_DIR
from aerotrack.runtime import resolve_device, synchronize
from aerotrack.ultra import configure, load_yolo, resolve_weights

STAGES = ("preprocess", "inference", "postprocess")


def export_onnx(weights: str | Path, imgsz: int, max_det: int) -> Path:
    """Static-shape batch-1 ONNX export, cached per (weights CONTENT, imgsz, max_det).

    Keyed by a hash of the checkpoint, not its name: every fine-tune is called `best.pt`,
    and a name-keyed cache once timed a 3-epoch YOLO26n export in place of a 100-epoch
    YOLO26s (DECISIONS.md)."""
    weights = resolve_weights(weights)
    if not Path(weights).exists():
        load_yolo(weights)  # a bare release name: download it before hashing
    digest = hashlib.sha256(Path(weights).read_bytes()).hexdigest()[:12]
    dest = WEIGHTS_DIR / "onnx" / f"{Path(weights).stem}_{digest}_{imgsz}_md{max_det}.onnx"
    if dest.exists():
        return dest
    model = load_yolo(weights)
    out = Path(model.export(format="onnx", imgsz=imgsz, batch=1, dynamic=False, simplify=True,
                            max_det=max_det, device="cpu", verbose=False))
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.rename(dest)
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
    ) -> None:
        configure()
        from ultralytics.data.augment import LetterBox  # noqa: PLC0415

        self.weights = resolve_weights(weights)
        self.backend = backend
        self.imgsz, self.conf, self.iou, self.max_det = imgsz, conf, iou, max_det
        self.letterbox = LetterBox(new_shape=(imgsz, imgsz), auto=False, stride=32)

        if backend == "torch":
            from ultralytics.nn.autobackend import AutoBackend  # noqa: PLC0415

            self.device = resolve_device(device)
            self.model = AutoBackend(self.weights, device=torch.device(self.device), fp16=half,
                                     fuse=True, verbose=False)
            self.model.eval()
            self.half = bool(self.model.fp16)
            self.end2end = bool(getattr(self.model, "end2end", False))
            head = getattr(getattr(self.model, "model", None), "model", [None])[-1]
            if self.end2end and hasattr(head, "max_det"):
                head.max_det = max_det  # top-k inside the NMS-free head
            self.names = dict(self.model.names)
            self.runtime = {"backend": "torch", "device": self.device, "half": self.half}
        elif backend == "onnxruntime":
            import onnxruntime as ort  # noqa: PLC0415

            if device not in ("auto", "cpu"):
                raise ValueError("the onnxruntime backend is benchmarked on CPU only")
            self.device = "cpu"
            self.half = False
            onnx_path = export_onnx(self.weights, imgsz, max_det)
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

    # -- stages ---------------------------------------------------------------------
    def preprocess(self, img_bgr: np.ndarray):
        padded = self.letterbox(image=img_bgr)
        chw = np.ascontiguousarray(padded[..., ::-1].transpose(2, 0, 1))[None]
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
        det = nms.non_max_suppression(raw, self.conf, self.iou, max_det=self.max_det,
                                      end2end=self.end2end)[0]
        det[:, :4] = ops.scale_boxes((self.imgsz, self.imgsz), det[:, :4], orig_shape)
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
