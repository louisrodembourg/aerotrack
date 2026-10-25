"""Device selection. Nothing in the repo may assume CUDA: the dev machine is Apple Silicon."""

import torch

VALID = ("auto", "cuda", "mps", "cpu")


def resolve_device(requested: str = "auto") -> str:
    """cuda > mps > cpu when `auto`; an explicit request that cannot be honoured is an
    error rather than a silent fallback, so a benchmark never reports the wrong device."""
    requested = requested.lower()
    requested_kind = "cuda" if requested.startswith("cuda:") else requested
    if requested_kind not in VALID:
        raise ValueError(f"unknown device {requested!r}, expected one of {VALID} or cuda:N")

    if requested_kind == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested_kind == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device 'cuda' requested but CUDA is not available")
    if requested_kind == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("device 'mps' requested but MPS is not available")
    return requested


def synchronize(device: str) -> None:
    """GPU kernels are asynchronous: without a sync, a timer measures the launch, not the work."""
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
