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


def _only(cfg: dict, splits: list[str] | None) -> dict:
    """The dataset config restricted to `splits` (all of them when None)."""
    if not splits:
        return cfg
    unknown = set(splits) - set(cfg["splits"])
    if unknown:
        raise KeyError(f"{cfg['name']}: unknown split(s) {sorted(unknown)}")
    return {**cfg, "splits": {k: v for k, v in cfg["splits"].items() if k in splits}}


def download(name: str, keep_archives: bool = True, splits: list[str] | None = None) -> Path:
    """Fetch and extract the dataset's archives; with `splits`, only the archives holding
    them (an archive is named after the split directory it extracts to)."""
    from aerotrack.data.download import Source, ensure_raw  # noqa: PLC0415

    cfg = dataset_config(name)
    sources = cfg["sources"]
    if splits:
        dirs = {v["dir"] for v in _only(cfg, splits)["splits"].values()}
        sources = [s for s in sources if Path(s["filename"]).stem in dirs]
        if len(sources) != len(dirs):
            raise KeyError(f"{name}: no archive per split for {splits}")
    return ensure_raw(name, [Source(**s) for s in sources], keep_archives=keep_archives)


def convert(name: str, splits: list[str] | None = None) -> dict:
    module = importlib.import_module(f"aerotrack.data.{name}")
    return module.convert(_only(dataset_config(name), splits))
