"""Single entry point to Ultralytics, so its global settings are pinned in one place."""

from pathlib import Path

from aerotrack.paths import DATA_DIR, RUNS_DIR, WEIGHTS_DIR

# Experiment tracking goes through aerotrack.experiments only. Ultralytics' own
# integrations would log a second, partial copy of each run (and `sync` sends usage
# analytics), so they are all switched off.
_SETTINGS = {
    "sync": False,
    "wandb": False,
    "mlflow": False,
    "clearml": False,
    "comet": False,
    "dvc": False,
    "raytune": False,
    "tensorboard": False,
    "vscode_msg": False,
    "openvino_msg": False,
}


def configure() -> None:
    from ultralytics.utils import SETTINGS  # noqa: PLC0415

    SETTINGS.update(
        {
            **_SETTINGS,
            "datasets_dir": str(DATA_DIR),
            "weights_dir": str(WEIGHTS_DIR),
            "runs_dir": str(RUNS_DIR / "ultralytics"),
        }
    )


def resolve_weights(weights: str | Path) -> str:
    """A bare release name (`yolo26n.pt`) is downloaded once into WEIGHTS_DIR instead of
    the current working directory, where Ultralytics would drop it by default."""
    p = Path(weights)
    if p.exists() or p.parent != Path(".") or p.suffix == ".yaml":
        return str(p)  # a bare .yaml is an architecture Ultralytics ships, not a download
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    return str(WEIGHTS_DIR / p.name)


def load_yolo(weights: str | Path):
    configure()
    from ultralytics import YOLO  # noqa: PLC0415

    return YOLO(resolve_weights(weights))
