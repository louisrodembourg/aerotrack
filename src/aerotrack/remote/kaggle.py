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
from aerotrack.experiments import find_train_run
from aerotrack.paths import REPO_ROOT, REPORTS_DIR, RUNS_DIR

KAGGLE_CLI = ["uvx", "--from", "kaggle==2.2.5", "kaggle"]
# Kaggle's free GPU (the P100 is retired; `kaggle kernels push --help`, CLI 2.2.5).
ACCELERATOR = "NvidiaTeslaT4"
KAGGLE_RUNS = "/kaggle/working/runs"
KAGGLE_REPORTS = "/kaggle/working/reports"
RESUME_DIR = "/tmp/aerotrack/resume"  # kaggle_kernel.py copies checkpoints there
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
    # Newer API tokens (~/.kaggle/access_token) do not store the username; the CLI resolves it.
    view = subprocess.run([*KAGGLE_CLI, "config", "view"], text=True, capture_output=True).stdout
    if (m := re.search(r"^- username: (\S+)$", view, re.MULTILINE)) and m.group(1) != "None":
        return m.group(1)
    raise SystemExit("Kaggle credentials not found: create an API token on kaggle.com "
                     "(Settings -> API -> Create New Token) and save it under ~/.kaggle/")


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
    """One job per config: a training config (configs/train) or an evaluation one
    (configs/eval), whose report is redirected into the kernel's saved output."""
    jobs = []
    for path in configs:
        cfg = load_config(path, overrides)
        name = f"{cfg['name']}-{suffix}" if suffix else cfg["name"]
        extra = [f"name={name}"] if suffix else []
        rel = Path(path).resolve().relative_to(REPO_ROOT)
        if "data" in cfg:
            jobs.append({"kind": "train", "config": str(rel), "overrides": [*overrides, *extra],
                         "dataset": cfg["data"]["dataset"], "name": name})
            continue
        ds = cfg["dataset"]
        dataset = ds["mot"] if "mot" in ds else ds["gt"].split("/")[0]
        jobs.append({"kind": "evaluate", "config": str(rel), "name": name, "dataset": dataset,
                     "overrides": [*overrides, *extra, f"report_dir={KAGGLE_REPORTS}/{name}"],
                     "inputs": eval_inputs(cfg)})
    return jobs


def upload_dataset(files: dict[str, Path], slug: str) -> str:
    """Upload local files as a private Kaggle dataset `<slug>`, wait until it is ready. File
    names carry their own key (run id): the kernel finds them by name under /kaggle/input,
    whatever directory Kaggle mounts the dataset in."""
    ref = kernel_ref(slug)
    folder = Path(tempfile.mkdtemp(prefix="aerotrack-dataset-"))
    for name, src in files.items():
        shutil.copy2(src, folder / name)
    (folder / "dataset-metadata.json").write_text(json.dumps(
        {"title": slug, "id": ref, "licenses": [{"name": "CC0-1.0"}]}))
    exists = subprocess.run([*KAGGLE_CLI, "datasets", "status", ref], capture_output=True,
                            text=True).returncode == 0
    if exists:  # a re-push of the same experiment: new version, same reference
        kaggle("datasets", "version", "-p", str(folder), "-m", "re-push", "-q")
    else:
        kaggle("datasets", "create", "-p", str(folder), "-q")  # private unless --public
    for _ in range(60):  # a kernel attached to a dataset still processing fails to start
        if "ready" in kaggle("datasets", "status", ref, capture=True):
            return ref
        time.sleep(10)
    raise SystemExit(f"dataset {ref} not ready after 10 min")


def upload_checkpoint(run_id: str, slug: str) -> str:
    """An interrupted run's last.pt / best.pt, so a new session can resume it."""
    weights = RUNS_DIR / "ultralytics" / run_id / "weights"
    if not (weights / "last.pt").exists():
        raise SystemExit(f"{weights}/last.pt not found: nothing to resume")
    return upload_dataset({f"{run_id}-{f}": weights / f for f in ("last.pt", "best.pt")
                           if (weights / f).exists()}, f"{slug}-ckpt")


def eval_inputs(cfg: dict) -> list[str]:
    """Training runs an evaluation config names (models and keyframe light detectors)."""
    refs = []
    for m in cfg.get("models", []):
        refs += [m["run"]] if "run" in m else []
        refs += [m["keyframe"]["light"]["run"]] if m.get("keyframe", {}).get("light") else []
    return list(dict.fromkeys(find_train_run(r)["id"] for r in refs))


def upload_runs(run_ids: list[str], slug: str) -> str:
    """Finished training runs (record + best.pt) for a Kaggle evaluation."""
    files = {}
    for rid in run_ids:
        rec = json.loads((RUNS_DIR / rid / "run.json").read_text())
        files[f"{rid}__run.json"] = RUNS_DIR / rid / "run.json"
        files[f"{rid}__best.pt"] = Path(rec["summary"]["weights"])
    return upload_dataset(files, f"{slug}-runs")


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
         dry_run: bool = False, resume: str | None = None) -> Path:
    if resume:
        if len(configs) != 1:
            raise SystemExit("--resume continues one run: pass its config alone")
        overrides = [*overrides, f"resume_from={RESUME_DIR}/{resume}"]
    jobs = make_jobs(configs, overrides, suffix)
    if len(jobs) > 2:
        raise SystemExit("a Kaggle session has 2 GPUs: at most 2 jobs per push")
    if resume:
        jobs[0]["resume"] = resume
    commit, b64 = bundle()
    datasets = [upload_checkpoint(resume, slug)] if resume and not dry_run else []
    inputs = sorted({r for j in jobs for r in j.get("inputs", [])})
    if inputs and not dry_run:
        datasets.append(upload_runs(inputs, slug))
    ref = kernel_ref(slug) if not dry_run else f"dry-run/{slug}"
    folder = Path(tempfile.mkdtemp(prefix="aerotrack-kaggle-"))
    (folder / "kernel.py").write_text(render(jobs, commit, b64))
    (folder / "kernel-metadata.json").write_text(json.dumps({
        "id": ref, "title": slug, "code_file": "kernel.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True,
        "enable_internet": True, "machine_shape": ACCELERATOR,
        "dataset_sources": datasets, "kernel_sources": [], "competition_sources": [],
    }, indent=2))
    record = {"kernel": ref, "commit": commit, "jobs": jobs, "accelerator": ACCELERATOR,
              "datasets": datasets,
              "pushed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if dry_run:
        print(f"dry run: kernel written to {folder}")
        return folder
    out = kaggle("kernels", "push", "-p", str(folder), capture=True)
    print(out.strip())
    # The CLI exits 0 even when Kaggle refuses the push (seen: "Kernel push error: Maximum
    # weekly GPU quota of 30.00 hours reached."); its message is the only signal.
    if "push error" in out.lower():
        raise SystemExit(f"Kaggle refused the push: {out.strip()}")
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


def merge_output(output: Path, slug: str, runs_dir: Path = RUNS_DIR,
                 reports_dir: Path = REPORTS_DIR) -> list[str]:
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
    src_reports = output / "reports"
    for d in sorted(src_reports.iterdir()) if src_reports.exists() else []:
        target = reports_dir / d.name
        if target.exists():
            raise SystemExit(f"report {target} already exists locally; not overwriting it")
        res = d / "results.json"
        if res.exists():
            res.write_text(json.dumps(relocate(json.loads(res.read_text()), KAGGLE_RUNS,
                                               str(runs_dir)), indent=2, default=str))
        shutil.move(str(d), target)
        merged.append(f"reports/{d.name}")
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
        # Seen once right after a kernel completed: the first download failed, the retry
        # worked. Output files can lag the COMPLETE status.
        for attempt in range(3):
            try:
                kaggle("kernels", "output", kernel_ref(slug), "-p", tmp)
                break
            except subprocess.CalledProcessError:
                if attempt == 2:
                    raise
                time.sleep(60)
        out = Path(tmp)
        # The CLI also saves the kernel's console log next to its output files.
        for log in out.glob("*.log"):
            (RUNS_DIR / "kaggle" / slug).mkdir(parents=True, exist_ok=True)
            shutil.copy(log, RUNS_DIR / "kaggle" / slug / log.name)
        merged = merge_output(out, slug)
    print(f"merged runs: {merged or 'none'}")
    return merged
