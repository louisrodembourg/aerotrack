"""Command-line entry point: `aerotrack <group> <command>` (see `aerotrack --help`)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False, pretty_exceptions_enable=False)
data_app = typer.Typer(no_args_is_help=True, help="Download, convert and check datasets.")
app.add_typer(data_app, name="data")

DatasetArg = Annotated[list[str], typer.Argument(help="visdrone_det | visdrone_mot | uavdt")]
SetOpt = Annotated[list[str] | None, typer.Option("--set", help="Override a config key: a.b=value")]


@data_app.command("download")
def data_download(
    datasets: DatasetArg,
    keep_archives: Annotated[bool, typer.Option(help="Keep verified archives after extraction")] = True,
) -> None:
    from aerotrack import data  # noqa: PLC0415

    for name in datasets:
        data.download(name, keep_archives=keep_archives)


@data_app.command("convert")
def data_convert(datasets: DatasetArg) -> None:
    from aerotrack import data  # noqa: PLC0415

    for name in datasets:
        data.convert(name)


@data_app.command("viz")
def data_viz(
    datasets: DatasetArg,
    n: Annotated[int, typer.Option(help="Samples per dataset")] = 12,
    seed: int = 0,
) -> None:
    """Draw converted labels on random frames, for a human check of every conversion."""
    from aerotrack.data.visualize import visualize  # noqa: PLC0415

    for name in datasets:
        out = visualize(name, n=n, seed=seed)
        typer.echo(f"{name}: {out}")


@data_app.command("check-trackeval")
def data_check_trackeval(
    dataset: Annotated[str, typer.Argument()] = "visdrone_mot",
    split: str = "val",
) -> None:
    """Feed the converted ground truth to TrackEval as if it were a tracker: every class must
    score HOTA = 1. Proves the files are readable by TrackEval and internally consistent."""
    from aerotrack.eval.tracking import gt_self_check  # noqa: PLC0415

    results = gt_self_check(dataset, split)
    for cls, hota in results.items():
        typer.echo(f"{dataset}/{split}/{cls}: HOTA={hota:.4f}")
    if any(abs(h - 1.0) > 1e-6 for h in results.values()):
        raise typer.Exit(1)


@app.command()
def train(
    config: Annotated[Path, typer.Argument(help="configs/train/*.yaml")],
    overrides: SetOpt = None,
) -> None:
    """Fine-tune a detector; tracked run (config, commit, seed, device, metrics, weights)."""
    from aerotrack.config import load_config  # noqa: PLC0415
    from aerotrack.train import run_training  # noqa: PLC0415

    run_training(load_config(config, overrides or []))


@app.command()
def evaluate(
    config: Annotated[Path, typer.Argument(help="configs/eval/*.yaml")],
    overrides: SetOpt = None,
) -> None:
    """Detection accuracy + latency for each model in the config; JSON + Markdown report."""
    from aerotrack.config import load_config  # noqa: PLC0415
    from aerotrack.eval.harness import run_evaluation  # noqa: PLC0415

    run_evaluation(load_config(config, overrides or []))


@app.command("readme-table")
def readme_table(
    readme: Path = Path("README.md"),
) -> None:
    """Regenerate the README results table from tracked reports only."""
    from aerotrack.eval.report import update_readme  # noqa: PLC0415

    update_readme(readme)


if __name__ == "__main__":
    app()
