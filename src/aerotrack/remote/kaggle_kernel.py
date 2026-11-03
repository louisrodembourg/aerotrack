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
  /kaggle/working/reports  evaluation reports (evaluation jobs)
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


def restore_inputs() -> list[str]:
    """Training runs an evaluation job needs (`kaggle push` attaches them as a dataset):
    record and best.pt back under /kaggle/working/runs, the weights path rewritten there,
    so `find_train_run` finds them exactly as on the machine that trained them."""
    ids = sorted({r for job in JOBS for r in job.get("inputs", [])})
    for rid in ids:
        rec_src = next(Path("/kaggle/input").rglob(f"{rid}__run.json"), None)
        pt_src = next(Path("/kaggle/input").rglob(f"{rid}__best.pt"), None)
        if rec_src is None or pt_src is None:
            raise SystemExit(f"input run {rid} not found under /kaggle/input")
        weights = OUT / "runs" / "ultralytics" / rid / "weights" / "best.pt"
        weights.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pt_src, weights)
        rec = json.loads(rec_src.read_text())
        rec["summary"]["weights"] = str(weights)
        (OUT / "runs" / rid).mkdir(parents=True, exist_ok=True)
        (OUT / "runs" / rid / "run.json").write_text(json.dumps(rec, indent=2))
    return ids


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
           "WANDB_PROJECT": os.environ.get("WANDB_PROJECT", "aerotrack"),
           # Private mirrors of archives the official host refuses (`kaggle mirror`).
           "AEROTRACK_MIRRORS": "/kaggle/input"}
    if key := wandb_key():
        env["WANDB_API_KEY"] = key
    else:
        env["WANDB_MODE"] = "offline"

    sh(["nvidia-smi"])
    print(f"vCPUs: {os.cpu_count()}", flush=True)
    # Same locked environment as everywhere else; Kaggle's preinstalled torch is not used.
    sh([sys.executable, "-m", "pip", "install", "-q", "uv==0.13.5"])
    uv = ["uv", "run", "--frozen", "--no-dev"]
    # --extra trt: TensorRT wheels for evaluation jobs (week 8); unused by training.
    sh(["uv", "sync", "--frozen", "--no-dev", "--extra", "trt"], cwd=REPO, env=env)
    uv = [*uv, "--extra", "trt"]

    if any(job.get("cpp") for job in JOBS):  # before the datasets: a build error fails fast
        # C++ runtime: TensorRT headers from NVIDIA's apt repository (preconfigured on
        # Kaggle) at the exact version of the wheel whose libraries built the engines.
        trt = subprocess.run([*uv, "python", "-c", "import tensorrt; print(tensorrt.__version__)"],
                             cwd=REPO, env=env, capture_output=True, text=True,
                             check=True).stdout.strip()
        sh(["apt-get", "update", "-q"])
        sh(["apt-get", "install", "-y", "-q", "--no-install-recommends",
            f"libnvinfer-headers-dev={trt}-1+cuda12.9"])
        sh([*uv, "aerotrack", "build-cpp", "--tensorrt"], cwd=REPO, env=env)

    for dataset in sorted({j["dataset"] for j in JOBS}):
        # Only the splits the jobs name; every split if one job needs them all (training).
        wanted = [j.get("splits") for j in JOBS if j["dataset"] == dataset]
        only = [] if None in wanted else sorted({s for w in wanted for s in w})
        opts = [x for s in only for x in ("--split", s)]
        sh([*uv, "aerotrack", "data", "download", dataset, "--no-keep-archives", *opts],
           cwd=REPO, env=env)
        sh([*uv, "aerotrack", "data", "convert", dataset, *opts], cwd=REPO, env=env)

    restored = restore_inputs()

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
        cmd = [*uv, "aerotrack", job.get("kind", "train"), job["config"]]
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
    for rid in restored:  # inputs, not results: keep them out of the saved output
        shutil.rmtree(OUT / "runs" / rid, ignore_errors=True)
        shutil.rmtree(OUT / "runs" / "ultralytics" / rid, ignore_errors=True)
    (OUT / "jobs.json").write_text(json.dumps(
        {"commit": GIT_COMMIT, "dirty": GIT_DIRTY, "jobs": status}, indent=2))
    return 0 if all(s["exit_code"] == 0 for s in status) else 1


if __name__ == "__main__":
    sys.exit(main())
