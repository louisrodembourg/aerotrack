import os
from pathlib import Path

import pytest

os.environ["WANDB_MODE"] = "disabled"

REAL_DATA = Path(__file__).resolve().parents[1] / "data"


def pytest_addoption(parser):
    parser.addoption("--run-network", action="store_true", help="run tests needing internet")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-network"):
        return
    skip = pytest.mark.skip(reason="needs --run-network")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)


def _snapshot(root: Path) -> set[tuple[str, int]]:
    if not root.exists():
        return set()
    return {(str(p), p.lstat().st_mtime_ns) for p in root.rglob("*") if p.is_file() and
            "_archives" not in p.parts}


@pytest.fixture(autouse=True)
def isolated_paths(tmp_path, monkeypatch):
    """Point every writable location to tmp, and fail if a test still touched data/."""
    import aerotrack.experiments
    import aerotrack.paths

    data = tmp_path / "data"
    monkeypatch.setattr(aerotrack.paths, "DATA_DIR", data)
    monkeypatch.setattr(aerotrack.experiments, "RUNS_DIR", tmp_path / "runs")
    processed = REAL_DATA / "processed"
    before = _snapshot(processed)
    yield data
    assert _snapshot(processed) == before, "a test modified data/processed"
