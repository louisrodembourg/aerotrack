"""YAML configs with dotted command-line overrides (`--set train.epochs=3`).

The resolved config (after overrides) is what gets logged with each run, so a run can be
replayed from its record alone.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _set(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
        if not isinstance(node, dict):
            raise KeyError(f"cannot override {dotted!r}: {k!r} is not a mapping")
    node[keys[-1]] = value


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    out = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        key, raw = item.split("=", 1)
        _set(out, key.strip(), yaml.safe_load(raw))  # YAML parsing: 3 -> int, [a,b] -> list
    return out


def load_config(path: Path | str, overrides: list[str] | None = None) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    cfg = apply_overrides(cfg, overrides or [])
    cfg["_source"] = str(path)
    return cfg
