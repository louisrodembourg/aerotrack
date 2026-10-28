"""Kaggle kernel template: runs on Kaggle, never imported locally.

`aerotrack kaggle push` fills the four placeholders below and uploads the result as a
script kernel. The repository travels inside the script (a base64 `git archive` of one
commit), so the kernel needs no dataset, no git access and no GitHub token, and the commit
it runs is exactly the one recorded in every run.

Layout on Kaggle:
  /tmp/aerotrack/repo      code (extracted bundle)
  /tmp/aerotrack/data      datasets, downloaded and converted here (not kept)
  /tmp/aerotrack/weights   pretrained checkpoints
  /tmp/aerotrack/resume    checkpoints of interrupted runs, from an attached dataset
  /kaggle/working/runs     run records + trained weights: the kernel's saved output
  /kaggle/working/logs     one log per job, GPU/CPU utilisation samples

Jobs run in parallel, one per GPU (Kaggle's free GPU is 2 x T4). Ultralytics DDP is not
used: it re-imports the trainer in a subprocess, which loses our masked dataset class and
the tracked-run callbacks (see DECISIONS.md).
"""

import base64
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

BUNDLE = "__BUNDLE__"
JOBS = json.loads("""__JOBS__""")
GIT_COMMIT = "__GIT_COMMIT__"
GIT_DIRTY = "__GIT_DIRTY__"

ROOT = Path("/tmp/aerotrack")
REPO = ROOT / "repo"
OUT = Path("/kaggle/working")
LOGS = OUT / "logs"
# Kaggle stops a GPU session at 12 h (reported, not found in Kaggle's own pages), and
# whether a killed session keeps its output is unknown. Jobs still running at this point
# are interrupted instead: their run is saved as "failed", with last.pt / best.pt intact.
DEADLINE_S = 11.25 * 3600
STARTED = time.monotonic()


def sh(cmd: list[str], **kw) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def wandb_key() -> str | None:
    # A Kaggle secret must be attached to the notebook once, in its web editor (Add-ons ->
    # Secrets); without it the runs stay offline and are synced later from the pulled output.
    try:
        from kaggle_secrets import UserSecretsClient  # noqa: PLC0415

        return UserSecretsClient().get_secret("WANDB_API_KEY")
    except Exception as e:  # noqa: BLE001
        print(f"no WANDB_API_KEY secret ({type(e).__name__}): W&B runs stay offline", flush=True)
        return None


def sample_utilisation(stop: threading.Event) -> None:
    """GPU and CPU load every 30 s: tells whether a run was GPU- or dataloader-bound,
    which decides how many jobs share the 4 vCPUs of a Kaggle session."""
    path = LOGS / "utilisation.csv"
    with path.open("w") as f:
        f.write("t_s,gpu,util_gpu_pct,mem_used_mib,load1\n")
        t0 = time.time()
        while not stop.wait(30):
            q = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True)
            load1 = os.getloadavg()[0]
            for line in q.stdout.strip().splitlines():
                idx, util, mem = (x.strip() for x in line.split(","))
                f.write(f"{time.time() - t0:.0f},{idx},{util},{mem},{load1:.2f}\n")
            f.flush()


def main() -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(BUNDLE)), mode="r:gz") as tar:
        try:
            tar.extractall(REPO, filter="data")
        except TypeError:  # Python without extraction filters; the bundle is our own archive
            tar.extractall(REPO)

    env = {**os.environ,
           "AEROTRACK_DATA": str(ROOT / "data"), "AEROTRACK_RUNS": str(OUT / "runs"),
           "AEROTRACK_WEIGHTS": str(ROOT / "weights"),
           "AEROTRACK_GIT_COMMIT": GIT_COMMIT, "AEROTRACK_GIT_DIRTY": GIT_DIRTY,
           "UV_PYTHON_INSTALL_DIR": str(ROOT / "python"), "UV_LINK_MODE": "copy",
           "WANDB_PROJECT": os.environ.get("WANDB_PROJECT", "aerotrack")}
    if key := wandb_key():
        env["WANDB_API_KEY"] = key
    else:
        env["WANDB_MODE"] = "offline"

    sh(["nvidia-smi"])
    print(f"vCPUs: {os.cpu_count()}", flush=True)
    # Same locked environment as everywhere else; Kaggle's preinstalled torch is not used.
    sh([sys.executable, "-m", "pip", "install", "-q", "uv==0.13.5"])
    uv = ["uv", "run", "--frozen", "--no-dev"]
    sh(["uv", "sync", "--frozen", "--no-dev"], cwd=REPO, env=env)

    for dataset in sorted({j["dataset"] for j in JOBS}):
        sh([*uv, "aerotrack", "data", "download", dataset, "--no-keep-archives"], cwd=REPO, env=env)
        sh([*uv, "aerotrack", "data", "convert", dataset], cwd=REPO, env=env)

    for job in JOBS:
        if run_id := job.get("resume"):
            # Checkpoint of an interrupted run, attached as a dataset by `kaggle push --resume`.
            dst = ROOT / "resume" / run_id
            dst.mkdir(parents=True, exist_ok=True)
            for src in Path("/kaggle/input").rglob(f"{run_id}-*.pt"):
                shutil.copy2(src, dst / src.name.removeprefix(f"{run_id}-"))
            if not (dst / "last.pt").exists():
                raise SystemExit(f"resume: {run_id}-last.pt not found under /kaggle/input")

    gpus = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.count("GPU ")
    if len(JOBS) > max(gpus, 1):
        raise SystemExit(f"{len(JOBS)} jobs for {gpus} GPU(s): one job per GPU")

    stop = threading.Event()
    threading.Thread(target=sample_utilisation, args=(stop,), daemon=True).start()
    procs = []
    for i, job in enumerate(JOBS):
        cmd = [*uv, "aerotrack", "train", job["config"]]
        for o in job["overrides"]:
            cmd += ["--set", o]
        log = (LOGS / f"job{i}-{Path(job['config']).stem}.log").open("w")
        print(f"job {i} on GPU {i}: {' '.join(cmd)}", flush=True)
        procs.append((job, log, subprocess.Popen(
            cmd, cwd=REPO, env={**env, "CUDA_VISIBLE_DEVICES": str(i)},
            stdout=log, stderr=subprocess.STDOUT)))

    status = []
    pending = list(procs)
    timed_out = set()
    while pending:
        for item in list(pending):
            job, log, p = item
            if p.poll() is None:
                continue
            pending.remove(item)
            log.close()
            status.append({**job, "exit_code": p.returncode, "log": log.name,
                           "interrupted_at_deadline": job["name"] in timed_out})
            print(f"--- {job['config']} exit {p.returncode}; last lines:", flush=True)
            print("".join(Path(log.name).read_text().splitlines(keepends=True)[-15:]), flush=True)
        if pending and time.monotonic() - STARTED > DEADLINE_S:
            for job, _, p in pending:
                if job["name"] not in timed_out:
                    print(f"deadline: interrupting {job['name']}", flush=True)
                    p.send_signal(signal.SIGINT)
                    timed_out.add(job["name"])
        time.sleep(10)
    stop.set()
    (OUT / "jobs.json").write_text(json.dumps(
        {"commit": GIT_COMMIT, "dirty": GIT_DIRTY, "jobs": status}, indent=2))
    return 0 if all(s["exit_code"] == 0 for s in status) else 1


if __name__ == "__main__":
    sys.exit(main())
