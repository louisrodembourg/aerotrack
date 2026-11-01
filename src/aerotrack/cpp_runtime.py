"""Build and load the C++ runtime (`cpp/`, Python module `aerotrack_cpp`).

Letterbox, TensorRT inference and NMS in C++, behind `Detector(backend="cpp")`. TensorRT is
built in only where the `tensorrt-cu12-libs` wheel is installed (Linux + NVIDIA); elsewhere
the module holds the host stages alone, which the tests compare with the Python ones.
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

from aerotrack.paths import REPO_ROOT

SOURCE_DIR = REPO_ROOT / "cpp"
BUILD_DIR = SOURCE_DIR / "build"
MODULE = "aerotrack_cpp"


def tensorrt_lib_dir() -> Path | None:
    """Directory of the wheel's libnvinfer.so.10: the library the engines were built with."""
    try:
        import tensorrt_libs  # noqa: PLC0415  # only present on the deployment target
    except ImportError:
        return None
    return Path(tensorrt_libs.__file__).parent


def build(build_dir: Path = BUILD_DIR, with_tensorrt: bool | None = None) -> Path:
    """Configure and compile with CMake; returns the build directory."""
    cmake = shutil.which("cmake")
    if cmake is None:
        raise RuntimeError("cmake not found (a dev dependency; or install it system-wide)")
    lib_dir = tensorrt_lib_dir()
    if with_tensorrt is None:
        with_tensorrt = lib_dir is not None
    if with_tensorrt and lib_dir is None:
        raise RuntimeError("TensorRT build requested but the tensorrt-cu12-libs wheel is missing")
    args = [cmake, "-S", str(SOURCE_DIR), "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DAEROTRACK_WITH_TRT={'ON' if with_tensorrt else 'OFF'}"]
    if with_tensorrt:
        args.append(f"-DTENSORRT_LIB_DIR={lib_dir}")
    subprocess.run(args, check=True)
    subprocess.run([cmake, "--build", str(build_dir), "-j"], check=True)
    return build_dir


def load(build_dir: Path = BUILD_DIR) -> ModuleType:
    if not any(build_dir.glob(f"{MODULE}*")):
        raise RuntimeError(f"C++ runtime not built in {build_dir}: run `aerotrack build-cpp`")
    if str(build_dir) not in sys.path:
        sys.path.insert(0, str(build_dir))
    return importlib.import_module(MODULE)
