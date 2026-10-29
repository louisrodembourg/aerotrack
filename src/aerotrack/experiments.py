"""Experiment tracking. Every run gets a local record (runs/<id>/run.json) AND a W&B run.

The local record is the source of truth for reports: README tables are generated from it,
so a number can always be traced back to a run id, a commit and a config. W&B is the
remote mirror that makes runs on a rented GPU visible without copying files back.

W&B mode: honours WANDB_MODE if set; otherwise `online` when credentials exist, else
`offline` (sync later with `wandb sync runs/wandb/offline-run-*`). Tests force `disabled`.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aerotrack.paths import RUNS_DIR
from aerotrack.runtime import environment_info, git_info
from aerotrack.runtime.git import git_diff

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "aerotrack")


def _wandb_mode() -> str:
    if "WANDB_MODE" in os.environ:
        return os.environ["WANDB_MODE"]
    if os.environ.get("WANDB_API_KEY"):
        return "online"
    netrc = Path.home() / ".netrc"
    if netrc.exists() and "api.wandb.ai" in netrc.read_text(errors="ignore"):
        return "online"
    return "offline"


def _jsonable(obj: Any) -> Any:
    return json.loads(json.dumps(obj, default=str))


@dataclass
class Run:
    id: str
    name: str
    job_type: str
    dir: Path
    record: dict
    _wandb: Any = field(default=None, repr=False)

    def log(self, metrics: dict, step: int | None = None) -> None:
        self.record.setdefault("history", []).append({"step": step, **_jsonable(metrics)})
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def summary(self, metrics: dict) -> None:
        """Final numbers of the run; these, and only these, feed reports and the README."""
        self.record.setdefault("summary", {}).update(_jsonable(metrics))
        if self._wandb is not None:
            self._wandb.summary.update(_flatten(metrics))

    def artifact(self, path: Path, kind: str = "file") -> None:
        path = Path(path)
        self.record.setdefault("artifacts", []).append({"path": str(path), "kind": kind})
        if self._wandb is not None and path.exists():
            import wandb  # noqa: PLC0415

            # W&B artifact names allow only [A-Za-z0-9_.-]; run names may carry others ("+").
            name = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{self.name}-{path.stem}")
            art = wandb.Artifact(name=name, type=kind)
            if path.is_dir():
                art.add_dir(str(path))
            else:
                art.add_file(str(path))
            self._wandb.log_artifact(art)

    def save(self) -> Path:
        out = self.dir / "run.json"
        out.write_text(json.dumps(self.record, indent=2, default=str))
        return out


def _flatten(d: dict, prefix: str = "") -> dict:
    flat = {}
    for k, v in d.items():
        key = f"{prefix}/{k}" if prefix else str(k)
        if isinstance(v, dict):
            flat.update(_flatten(v, key))
        elif isinstance(v, int | float | str | bool) or v is None:
            flat[key] = v
    return flat


@contextmanager
def start_run(
    name: str,
    job_type: str,
    config: dict,
    seed: int | None,
    device: str | None,
    tags: tuple[str, ...] = (),
):
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{name}-{uuid.uuid4().hex[:6]}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    git = git_info()
    record = {
        "id": run_id,
        "name": name,
        "job_type": job_type,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seed": seed,
        "device": device,
        "git": git,
        "env": environment_info(device),
        "config": _jsonable(config),
        "status": "running",
    }
    if git.get("dirty"):
        (run_dir / "uncommitted.diff").write_text(git_diff())

    mode = _wandb_mode()
    wb = None
    if mode != "disabled":
        import wandb  # noqa: PLC0415

        wb = wandb.init(
            project=WANDB_PROJECT,
            name=run_id,
            job_type=job_type,
            tags=list(tags),
            config={**record["config"], "_seed": seed, "_device": device, "_git": git,
                    "_env": record["env"]},
            dir=str(RUNS_DIR),
            mode=mode,
        )
        record["wandb"] = {"mode": mode, "url": getattr(wb, "url", None)}

    run = Run(id=run_id, name=name, job_type=job_type, dir=run_dir, record=record, _wandb=wb)
    try:
        yield run
        record["status"] = "finished"
    except BaseException:
        record["status"] = "failed"
        raise
    finally:
        record["ended"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        run.save()
        if wb is not None:
            wb.finish(exit_code=0 if record["status"] == "finished" else 1)


def find_train_run(ref: str) -> dict:
    """Finished training run by id, or by name when exactly one finished run carries it.
    Two runs with the same name are an error rather than "take the latest": an evaluation
    config must say which training it scores."""
    records = [json.loads(p.read_text()) for p in sorted(RUNS_DIR.glob("*/run.json"))]
    finished = [r for r in records if r.get("job_type") == "train" and r.get("status") == "finished"]
    if match := [r for r in finished if r["id"] == ref]:
        return match[0]
    match = [r for r in finished if r["name"] == ref]
    if len(match) == 1:
        return match[0]
    if not match:
        raise LookupError(f"no finished training run with id or name {ref!r} in {RUNS_DIR}")
    raise LookupError(f"{len(match)} finished training runs named {ref!r}; use a run id: "
                      + ", ".join(r["id"] for r in match))
