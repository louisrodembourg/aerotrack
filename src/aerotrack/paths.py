"""Filesystem layout. Every location can be overridden by an environment variable so the
same code runs on the Mac, in the CUDA container and on a rented GPU box."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_path(var: str, default: Path) -> Path:
    return Path(os.environ[var]).expanduser().resolve() if var in os.environ else default


DATA_DIR = _env_path("AEROTRACK_DATA", REPO_ROOT / "data")
RAW_DIR = DATA_DIR / "raw"
ARCHIVES_DIR = RAW_DIR / "_archives"
PROCESSED_DIR = DATA_DIR / "processed"
RUNS_DIR = _env_path("AEROTRACK_RUNS", REPO_ROOT / "runs")
REPORTS_DIR = REPO_ROOT / "reports"
WEIGHTS_DIR = _env_path("AEROTRACK_WEIGHTS", REPO_ROOT / "weights")
CONFIGS_DIR = REPO_ROOT / "configs"


def data_relative(path: Path) -> str:
    """Store dataset paths relative to DATA_DIR so annotation files stay valid when the
    data directory is mounted somewhere else (container, remote box)."""
    return Path(path).resolve().relative_to(DATA_DIR.resolve()).as_posix()


def from_data_relative(rel: str) -> Path:
    return DATA_DIR / rel
