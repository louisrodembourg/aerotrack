"""C++ runtime host stages (cpp/) against the Python Detector's.

The module is compiled once per session into tmp, without TensorRT (which never runs on
the Mac); TrtDetector itself is checked on the T4, through the evaluation harness.
"""

import shutil

import numpy as np
import pytest
import torch

from aerotrack.inference import Detector

pytestmark = pytest.mark.skipif(shutil.which("cmake") is None, reason="needs cmake")


@pytest.fixture(scope="session")
def cpp(tmp_path_factory):
    from aerotrack import cpp_runtime

    build = cpp_runtime.build(tmp_path_factory.mktemp("cpp-build"), with_tensorrt=False)
    return cpp_runtime.load(build)


def _python_postprocess(raw: np.ndarray, imgsz: int, shape, conf=0.001, iou=0.7, max_det=300):
    """Detector.postprocess on a raw head output, without loading a model."""
    d = Detector.__new__(Detector)
    d.imgsz, d.conf, d.iou, d.max_det, d.merge_ios = imgsz, conf, iou, max_det, 0.5
    d.backend, d.end2end, d.host_nms, d.device = "torch", False, True, "cpu"
    d._crops = [(0, 0, shape[1], shape[0])]
    return d.postprocess(torch.from_numpy(raw[None]), shape)


def _assert_same_detections(out: np.ndarray, ref: np.ndarray, atol: float) -> None:
    """Same rows in any order: torch's argsort is not stable, so equal scores may swap."""
    assert out.shape == ref.shape
    out, ref = out[np.lexsort(out[:, [0, 5, 4]].T)], ref[np.lexsort(ref[:, [0, 5, 4]].T)]
    np.testing.assert_array_equal(out[:, 5], ref[:, 5])
    np.testing.assert_allclose(out[:, :5], ref[:, :5], atol=atol)


@pytest.mark.parametrize(("h", "w", "size"), [
    (765, 1360, 640),    # VisDrone frame, downscaled
    (540, 960, 1280),    # upscaled
    (1280, 1280, 640),   # exact 2x
    (640, 640, 640),     # no resize, no padding
    (37, 1000, 640),     # thin image, most of it padding
])
def test_letterbox_matches_ultralytics(cpp, h, w, size):
    """Same geometry as Ultralytics' LetterBox; pixels bit-exact with cv2's
    INTER_LINEAR_EXACT, within 1 of its INTER_LINEAR (cpp/src/pipeline.hpp)."""
    import cv2
    from ultralytics.data.augment import LetterBox

    img = np.random.default_rng(h * w).integers(0, 256, (h, w, 3), dtype=np.uint8)
    out = cpp.letterbox(img, size)
    ref = LetterBox(new_shape=(size, size), auto=False, stride=32)(image=img)
    assert out.shape == ref.shape
    assert np.abs(out.astype(int) - ref.astype(int)).max() <= 1
    exact = LetterBox(new_shape=(size, size), auto=False, stride=32,
                      interpolation=cv2.INTER_LINEAR_EXACT)(image=img)
    # OpenCV's exact variant computes source coordinates in fixed point, ours in float:
    # a rare rounding apart.
    assert (out != exact).mean() < 0.005


@pytest.mark.parametrize("nms", ["torchvision", "ultralytics"])
def test_postprocess_matches_python_on_clustered_boxes(cpp, nms, monkeypatch):
    """Many overlapping candidates per object, of several classes: NMS decides a lot.
    Ultralytics runs torchvision's NMS once torchvision is imported, its own otherwise."""
    import sys

    import torchvision  # noqa: F401

    if nms == "ultralytics":
        monkeypatch.delitem(sys.modules, "torchvision")
    rng = np.random.default_rng(0)
    nc, n, size, shape = 10, 8400, 640, (765, 1360)
    centers = rng.uniform(20, 620, (150, 2))
    owner = rng.integers(0, len(centers), n)
    raw = np.empty((4 + nc, n), np.float32)
    raw[:2] = (centers[owner] + rng.normal(0, 3, (n, 2))).T
    raw[2:4] = rng.uniform(4, 60, (2, n))
    raw[4:] = rng.uniform(0, 1, (nc, n)) ** 6
    ref = _python_postprocess(raw, size, shape)
    out = cpp.postprocess(raw, size, *shape, 0.001, 0.7, 300)
    assert len(ref) > 100
    _assert_same_detections(out, ref, atol=1e-3)


def test_postprocess_caps_candidates_and_detections(cpp):
    """1280 input: 33600 anchors, above the 30000 candidates Ultralytics keeps."""
    rng = np.random.default_rng(1)
    nc, n, size, shape = 10, 33600, 1280, (1080, 1920)
    raw = np.empty((4 + nc, n), np.float32)
    raw[:2] = rng.uniform(0, size, (2, n))
    raw[2:4] = rng.uniform(2, 30, (2, n))
    raw[4:] = rng.uniform(0.01, 1, (nc, n))
    ref = _python_postprocess(raw, size, shape, max_det=500)
    out = cpp.postprocess(raw, size, *shape, 0.001, 0.7, 500)
    assert len(ref) == 500
    _assert_same_detections(out, ref, atol=1e-3)


def test_cpp_stages_around_torch_match_the_python_pipeline(cpp):
    """A real model: C++ letterbox -> torch inference -> C++ NMS gives the Python detections."""
    import cv2
    from ultralytics.utils import ASSETS

    img = cv2.imread(str(ASSETS / "bus.jpg"))
    d = Detector("yolo26n.pt", device="cpu", conf=0.25)
    ref = d(img)
    padded = cpp.letterbox(img, d.imgsz)
    chw = np.ascontiguousarray(padded[..., ::-1].transpose(2, 0, 1))[None]
    raw = d.infer(torch.from_numpy(chw).float().div_(255))
    raw = (raw[0] if isinstance(raw, list | tuple) else raw)[0].numpy()
    out = cpp.postprocess(raw, d.imgsz, *img.shape[:2], d.conf, d.iou, d.max_det)
    assert len(ref) > 0
    # Pixels within 1 of cv2's INTER_LINEAR move a box by about a pixel.
    _assert_same_detections(out, ref, atol=2.0)
