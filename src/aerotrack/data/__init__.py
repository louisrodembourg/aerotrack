"""Dataset registry: name -> config file and converter."""

from __future__ import annotations

import importlib
from pathlib import Path

import yaml

from aerotrack.paths import CONFIGS_DIR

DATASETS = ("visdrone_det", "visdrone_mot", "uavdt")


def dataset_config(name: str) -> dict:
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}, expected one of {DATASETS}")
    return yaml.safe_load(Path(CONFIGS_DIR / "data" / f"{name}.yaml").read_text())


def download(name: str, keep_archives: bool = True) -> Path:
    from aerotrack.data.download import Source, ensure_raw  # noqa: PLC0415

    cfg = dataset_config(name)
    return ensure_raw(name, [Source(**s) for s in cfg["sources"]], keep_archives=keep_archives)


def convert(name: str) -> dict:
    module = importlib.import_module(f"aerotrack.data.{name}")
    return module.convert(dataset_config(name))
