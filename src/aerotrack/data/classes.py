"""Explicit class maps between dataset vocabularies (configs/data/class_maps.yaml)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import yaml

from aerotrack.paths import CONFIGS_DIR

CLASS_MAPS_FILE = CONFIGS_DIR / "data" / "class_maps.yaml"


@dataclass(frozen=True)
class ClassMap:
    name: str
    source: str
    vocabulary: dict[int, str]
    mapping: dict[int, str]
    excluded: dict[int, str]
    ignore: tuple[int, ...]
    # Model class NAME -> target, for model classes the target vocabulary lacks (VisDrone
    # `van` scored on UAVDT, whose vans are labelled `car`). Several model classes then
    # feed one target, and `merge_nms` suppresses their duplicates (DECISIONS.md).
    predictions: tuple[tuple[str, str], ...] = ()
    merge_nms: bool = False

    @property
    def targets(self) -> list[str]:
        """Target classes in first-appearance order (stable category ids for COCO)."""
        seen: dict[str, None] = {}
        for sid in sorted(self.mapping):
            seen.setdefault(self.mapping[sid], None)
        return list(seen)

    def role(self, source_id: int) -> str:
        if source_id in self.mapping:
            return "mapped"
        if source_id in self.excluded:
            return "excluded"
        if source_id in self.ignore:
            return "ignore"
        raise KeyError(f"class map {self.name!r}: source id {source_id} is not accounted for")

    def validate(self) -> None:
        sets = {"mapping": set(self.mapping), "excluded": set(self.excluded),
                "ignore": set(self.ignore)}
        names = list(sets)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if overlap := sets[a] & sets[b]:
                    raise ValueError(f"class map {self.name!r}: ids {overlap} in both {a} and {b}")
        covered = set().union(*sets.values())
        vocab = set(self.vocabulary)
        if covered != vocab:
            raise ValueError(
                f"class map {self.name!r}: unaccounted ids {vocab - covered}, unknown ids {covered - vocab}"
            )


@cache
def _load(path: Path = CLASS_MAPS_FILE) -> dict:
    return yaml.safe_load(path.read_text())


def vocabulary(name: str) -> dict[int, str]:
    return {int(k): v for k, v in _load()["vocabularies"][name].items()}


def get_class_map(name: str) -> ClassMap:
    raw = _load()["maps"][name]
    cm = ClassMap(
        name=name,
        source=raw["source"],
        vocabulary=vocabulary(raw["source"]),
        mapping={int(k): v for k, v in raw["mapping"].items()},
        excluded={int(k): v for k, v in (raw.get("excluded") or {}).items()},
        ignore=tuple(int(x) for x in raw.get("ignore") or ()),
        predictions=tuple((raw.get("predictions") or {}).items()),
        merge_nms=bool(raw.get("merge_nms", False)),
    )
    cm.validate()
    return cm


def all_class_maps() -> list[str]:
    return list(_load()["maps"])
