"""Command-line entry point: `aerotrack <group> <command>` (see `aerotrack --help`)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False, pretty_exceptions_enable=False)
data_app = typer.Typer(no_args_is_help=True, help="Download, convert and check datasets.")
app.add_typer(data_app, name="data")

DatasetArg = Annotated[list[str], typer.Argument(help="visdrone_det | visdrone_mot | uavdt")]
SplitOpt = Annotated[list[str] | None, typer.Option(
    "--split", help="Only this split (repeatable); default: every split")]
SetOpt = Annotated[list[str] | None, typer.Option("--set", help="Override a config key: a.b=value")]


@data_app.command("download")
def data_download(
    datasets: DatasetArg,
    keep_archives: Annotated[bool, typer.Option(help="Keep verified archives after extraction")] = True,
    split: SplitOpt = None,
) -> None:
    from aerotrack import data  # noqa: PLC0415

    for name in datasets:
        data.download(name, keep_archives=keep_archives, splits=split)


@data_app.command("convert")
def data_convert(datasets: DatasetArg, split: SplitOpt = None) -> None:
    from aerotrack import data  # noqa: PLC0415

    for name in datasets:
        data.convert(name, splits=split)


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
    """Accuracy + latency for each model in the config (detection, or tracking when the
    config has a `tracker` section); JSON + Markdown report."""
    from aerotrack.config import load_config  # noqa: PLC0415

    cfg = load_config(config, overrides or [])
    if "tracker" in cfg:
        from aerotrack.eval.track_harness import run_tracking_evaluation  # noqa: PLC0415

        run_tracking_evaluation(cfg)
    else:
        from aerotrack.eval.harness import run_evaluation  # noqa: PLC0415

        run_evaluation(cfg)


@app.command("tune-tracker")
def tune_tracker(
    config: Annotated[Path, typer.Argument(help="configs/tune/*.yaml")],
    overrides: SetOpt = None,
) -> None:
    """Grid-search tracker thresholds on MOT sequences unseen by the detector."""
    from aerotrack.config import load_config  # noqa: PLC0415
    from aerotrack.eval.tune_tracker import run_tuning  # noqa: PLC0415

    run_tuning(load_config(config, overrides or []))


@app.command("demo-export")
def demo_export(
    config: Annotated[Path, typer.Argument(help="configs/demo/*.yaml")],
    overrides: SetOpt = None,
) -> None:
    """Track the demo clips and gather the headline numbers -> demo/public/ (Remotion)."""
    from aerotrack.config import load_config  # noqa: PLC0415
    from aerotrack.demo import export_demo  # noqa: PLC0415

    export_demo(load_config(config, overrides or []))


@app.command("build-cpp")
def build_cpp(
    tensorrt: Annotated[bool | None, typer.Option(
        help="Build the TensorRT runtime (default: when the tensorrt-cu12-libs wheel is there)")] = None,
) -> None:
    """Compile the C++ runtime (cpp/) into cpp/build/, for Detector(backend="cpp")."""
    from aerotrack import cpp_runtime  # noqa: PLC0415

    cpp = cpp_runtime.load(cpp_runtime.build(with_tensorrt=tensorrt))
    typer.echo(f"aerotrack_cpp: {cpp.compiler}, TensorRT {'on' if cpp.with_tensorrt else 'off'}")


kaggle_app = typer.Typer(no_args_is_help=True, help="Train on Kaggle's free GPUs (push, status, pull).")
app.add_typer(kaggle_app, name="kaggle")


@kaggle_app.command("push")
def kaggle_push(
    configs: Annotated[list[Path], typer.Argument(help="1 or 2 configs/train/*.yaml, one per GPU")],
    slug: Annotated[str, typer.Option(help="Kaggle kernel slug (one kernel per experiment)")],
    overrides: SetOpt = None,
    suffix: Annotated[str | None, typer.Option(help="Appended to each run name")] = None,
    dry_run: Annotated[bool, typer.Option(help="Write the kernel locally, do not upload")] = False,
    resume: Annotated[str | None, typer.Option(
        help="Run id of an interrupted run of this config: continue it from its last epoch")] = None,
) -> None:
    """Upload HEAD and run the training configs on a Kaggle GPU session."""
    from aerotrack.remote import kaggle  # noqa: PLC0415

    kaggle.push(configs, overrides or [], suffix, slug, dry_run=dry_run, resume=resume)


@kaggle_app.command("mirror")
def kaggle_mirror(
    dataset: Annotated[str, typer.Argument(help="visdrone_det | visdrone_mot | uavdt")],
    split: SplitOpt = None,
) -> None:
    """Upload archives rebuilt from data/raw as private Kaggle datasets, for when the
    official host refuses (Google Drive quota). Pin the printed content_sha256 in
    configs/data/<dataset>.yaml: kernels use a mirror only if its content matches."""
    from aerotrack.remote import kaggle  # noqa: PLC0415

    for ref in kaggle.upload_mirror(dataset, split or []):
        typer.echo(f"mirror ready: {ref}")


@kaggle_app.command("status")
def kaggle_status(slug: str) -> None:
    from aerotrack.remote import kaggle  # noqa: PLC0415

    typer.echo(kaggle.status(slug))


@kaggle_app.command("pull")
def kaggle_pull(slug: str) -> None:
    """Download the kernel's output and merge its run records into runs/."""
    from aerotrack.remote import kaggle  # noqa: PLC0415

    kaggle.pull(slug)


@app.command("readme-table")
def readme_table(
    readme: Path = Path("README.md"),
) -> None:
    """Regenerate the README results table from tracked reports only."""
    from aerotrack.eval.report import update_readme  # noqa: PLC0415

    update_readme(readme)


if __name__ == "__main__":
    app()
