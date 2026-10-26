"""Hardware and software fingerprint attached to every run and every latency result.
A latency number without device, runtime versions and (on Jetson) power mode is not
comparable to anything, so this is collected unconditionally."""

import importlib.metadata
import platform
import shutil
import subprocess
from pathlib import Path

import torch


def _version(dist: str) -> str | None:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _cpu_name() -> str:
    if platform.system() == "Darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def _jetson_power_mode() -> str | None:
    if not Path("/etc/nv_tegra_release").exists() or not shutil.which("nvpmodel"):
        return None
    return _run(["nvpmodel", "-q"])


def _macos_power() -> dict | None:
    """Power source and Low Power Mode on macOS: both can throttle the M-series CPU and
    GPU, so they belong to the latency context like the Jetson power mode."""
    if platform.system() != "Darwin" or not shutil.which("pmset"):
        return None
    batt = _run(["pmset", "-g", "batt"]) or ""
    settings = _run(["pmset", "-g"]) or ""
    low = next((line.split()[-1] for line in settings.splitlines()
                if line.strip().startswith("lowpowermode")), None)
    source = "ac" if "AC Power" in batt else "battery" if "Battery Power" in batt else None
    return {"source": source, "low_power_mode": low == "1" if low is not None else None}


def _tensorrt_version() -> str | None:
    try:
        import tensorrt  # noqa: PLC0415  # only present on the deployment target

        return tensorrt.__version__
    except ImportError:
        return None


def device_name(device: str) -> str:
    if device.startswith("cuda"):
        return torch.cuda.get_device_name(torch.device(device))
    if device == "mps":
        return f"{_cpu_name()} GPU (MPS)"
    return _cpu_name()


def environment_info(device: str | None = None) -> dict:
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": _cpu_name(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "ultralytics": _version("ultralytics"),
        "onnxruntime": _version("onnxruntime") or _version("onnxruntime-gpu"),
        "tensorrt": _tensorrt_version(),
        "jetson_power_mode": _jetson_power_mode(),
        "macos_power": _macos_power(),
    }
    if device is not None:
        info["device"] = device
        info["device_name"] = device_name(device)
    return info
