"""Train on Kaggle's free GPUs from the local machine: push, status, pull.

`push` renders `kaggle_kernel.py` with a `git archive` of HEAD and the jobs, then uploads it
as a private GPU script kernel. `pull` downloads the kernel's output and merges its run
records into the local `runs/`, so a Kaggle run is reported exactly like a local one.

The Kaggle CLI is called through `uvx` at a pinned version rather than added to the
lockfile: it is a local launcher, never needed by the training environment itself.
"""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from aerotrack.config import load_config
from aerotrack.paths import REPO_ROOT, RUNS_DIR

KAGGLE_CLI = ["uvx", "--from", "kaggle==2.2.5", "kaggle"]
# Kaggle's free GPU (the P100 is retired; `kaggle kernels push --help`, CLI 2.2.5).
ACCELERATOR = "NvidiaTeslaT4"
KAGGLE_RUNS = "/kaggle/working/runs"
BUNDLE_PATHS = ["pyproject.toml", "uv.lock", ".python-version", "README.md", "src", "configs", "scripts"]
TEMPLATE = Path(__file__).with_name("kaggle_kernel.py")


def kaggle(*args: str, capture: bool = False) -> str:
    out = subprocess.run([*KAGGLE_CLI, *args], check=True, text=True,
                         capture_output=capture)
    return out.stdout if capture else ""


def username() -> str:
    if user := os.environ.get("KAGGLE_USERNAME"):
        return user
    cred = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"
    if cred.exists():
        return json.loads(cred.read_text())["username"]
    raise SystemExit("Kaggle credentials not found: create an API token on kaggle.com "
                     "(Settings -> API -> Create New Token) and save it as ~/.kaggle/kaggle.json")


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args], check=True, text=True,
                          capture_output=True).stdout.strip()


def bundle() -> tuple[str, str]:
    """Base64 tar.gz of HEAD. Uncommitted changes are refused rather than silently left
    behind: the kernel runs the commit, and only the commit."""
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("working tree has uncommitted changes: commit them first, the "
                         "kernel runs HEAD exactly")
    commit = _git("rev-parse", "HEAD")
    data = subprocess.run(["git", "-C", str(REPO_ROOT), "archive", "--format=tar.gz", commit,
                           *BUNDLE_PATHS], check=True, capture_output=True).stdout
    return commit, base64.b64encode(data).decode()


def make_jobs(configs: list[Path], overrides: list[str], suffix: str | None) -> list[dict]:
    jobs = []
    for path in configs:
        cfg = load_config(path, overrides)
        extra = [f"name={cfg['name']}-{suffix}"] if suffix else []
        rel = Path(path).resolve().relative_to(REPO_ROOT)
        jobs.append({"config": str(rel), "overrides": [*overrides, *extra],
                     "dataset": cfg["data"]["dataset"], "name": f"{cfg['name']}-{suffix}"
                     if suffix else cfg["name"]})
    return jobs


def render(jobs: list[dict], commit: str, bundle_b64: str) -> str:
    src = TEMPLATE.read_text()
    for key, value in {"__JOBS__": json.dumps(jobs), "__GIT_COMMIT__": commit,
                       "__GIT_DIRTY__": "false", "__BUNDLE__": bundle_b64}.items():
        if src.count(key) != 1:
            raise RuntimeError(f"template placeholder {key} must appear exactly once")
        src = src.replace(key, value)
    return src


def kernel_ref(slug: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{3,48}", slug):
        raise SystemExit(f"invalid kernel slug {slug!r}: lowercase letters, digits, dashes")
    return f"{username()}/{slug}"


def push(configs: list[Path], overrides: list[str], suffix: str | None, slug: str,
         dry_run: bool = False) -> Path:
    jobs = make_jobs(configs, overrides, suffix)
    if len(jobs) > 2:
        raise SystemExit("a Kaggle session has 2 GPUs: at most 2 jobs per push")
    commit, b64 = bundle()
    ref = kernel_ref(slug) if not dry_run else f"dry-run/{slug}"
    folder = Path(tempfile.mkdtemp(prefix="aerotrack-kaggle-"))
    (folder / "kernel.py").write_text(render(jobs, commit, b64))
    (folder / "kernel-metadata.json").write_text(json.dumps({
        "id": ref, "title": slug, "code_file": "kernel.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True,
        "enable_internet": True, "machine_shape": ACCELERATOR,
        "dataset_sources": [], "kernel_sources": [], "competition_sources": [],
    }, indent=2))
    record = {"kernel": ref, "commit": commit, "jobs": jobs, "accelerator": ACCELERATOR,
              "pushed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if dry_run:
        print(f"dry run: kernel written to {folder}")
        return folder
    kaggle("kernels", "push", "-p", str(folder))
    log = RUNS_DIR / "kaggle" / slug
    log.mkdir(parents=True, exist_ok=True)
    (log / f"push-{time.strftime('%Y%m%d-%H%M%S')}.json").write_text(json.dumps(record, indent=2))
    print(f"pushed {ref} (commit {commit[:7]}); follow with `aerotrack kaggle status {slug}`")
    return folder


def status(slug: str) -> str:
    return kaggle("kernels", "status", kernel_ref(slug), capture=True).strip()


def relocate(obj, old: str, new: str):
    if isinstance(obj, str):
        return new + obj[len(old):] if obj.startswith(old) else obj
    if isinstance(obj, list):
        return [relocate(x, old, new) for x in obj]
    if isinstance(obj, dict):
        return {k: relocate(v, old, new) for k, v in obj.items()}
    return obj


def merge_output(output: Path, slug: str, runs_dir: Path = RUNS_DIR) -> list[str]:
    """Move pulled run records into `runs_dir`. Paths recorded on Kaggle are rewritten to
    their local location, and the rewrite itself is recorded in the run."""
    merged = []
    src_runs = output / "runs"
    for d in sorted(src_runs.iterdir()) if src_runs.exists() else []:
        if d.name in ("ultralytics", "wandb"):
            continue
        if (runs_dir / d.name).exists():
            raise SystemExit(f"run {d.name} already exists locally; not overwriting it")
    for d in sorted(src_runs.iterdir()) if src_runs.exists() else []:
        if d.name in ("ultralytics", "wandb"):
            for sub in d.iterdir():
                target = runs_dir / d.name / sub.name
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(sub), target)
            continue
        run_json = d / "run.json"
        if run_json.exists():
            rec = relocate(json.loads(run_json.read_text()), KAGGLE_RUNS, str(runs_dir))
            rec["relocated"] = {"from": KAGGLE_RUNS, "to": str(runs_dir), "kernel": slug}
            run_json.write_text(json.dumps(rec, indent=2, default=str))
        shutil.move(str(d), runs_dir / d.name)
        merged.append(d.name)
    meta = runs_dir / "kaggle" / slug
    meta.mkdir(parents=True, exist_ok=True)
    for name in ("logs", "jobs.json"):
        if (output / name).exists():
            target = meta / name
            if target.exists():
                shutil.rmtree(target) if target.is_dir() else target.unlink()
            shutil.move(str(output / name), target)
    return merged


def pull(slug: str) -> list[str]:
    with tempfile.TemporaryDirectory(prefix="aerotrack-pull-") as tmp:
        kaggle("kernels", "output", kernel_ref(slug), "-p", tmp)
        out = Path(tmp)
        # The CLI also saves the kernel's console log next to its output files.
        for log in out.glob("*.log"):
            (RUNS_DIR / "kaggle" / slug).mkdir(parents=True, exist_ok=True)
            shutil.copy(log, RUNS_DIR / "kaggle" / slug / log.name)
        merged = merge_output(out, slug)
    print(f"merged runs: {merged or 'none'}")
    return merged
