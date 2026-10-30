"""Kaggle launcher: what can be checked without an account or a GPU."""

import base64
import io
import json
import tarfile

from aerotrack.paths import CONFIGS_DIR
from aerotrack.remote import kaggle


def _bundle_b64() -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        data = b"print('hi')\n"
        info = tarfile.TarInfo("src/aerotrack/__init__.py")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return base64.b64encode(buf.getvalue()).decode()


def test_rendered_kernel_compiles_and_carries_jobs():
    jobs = kaggle.make_jobs([CONFIGS_DIR / "train" / "smoke.yaml"], ["train.epochs=1"], "probe")
    assert jobs == [{"kind": "train", "config": "configs/train/smoke.yaml",
                     "overrides": ["train.epochs=1", "name=smoke-probe"],
                     "dataset": "visdrone_det", "name": "smoke-probe"}]
    src = kaggle.render(jobs, "a" * 40, _bundle_b64())
    assert "__BUNDLE__" not in src and "__JOBS__" not in src
    compile(src, "kernel.py", "exec")
    ns: dict = {}
    exec(src.split("\nROOT = ")[0], ns)  # constants only: the body needs Kaggle
    assert ns["JOBS"] == jobs and ns["GIT_COMMIT"] == "a" * 40
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(ns["BUNDLE"])), mode="r:gz") as tar:
        assert tar.getnames() == ["src/aerotrack/__init__.py"]


def test_merge_output_relocates_paths_and_never_overwrites(tmp_path):
    out, runs = tmp_path / "out", tmp_path / "runs"
    rid = "20261026-000000-visdrone-yolo26n-640-abcdef"
    (out / "runs" / rid).mkdir(parents=True)
    (out / "runs" / "ultralytics" / rid / "weights").mkdir(parents=True)
    (out / "runs" / "ultralytics" / rid / "weights" / "best.pt").write_bytes(b"w")
    (out / "logs").mkdir()
    (out / "logs" / "utilisation.csv").write_text("t_s\n")
    best = f"{kaggle.KAGGLE_RUNS}/ultralytics/{rid}/weights/best.pt"
    (out / "runs" / rid / "run.json").write_text(json.dumps(
        {"summary": {"weights": best}, "artifacts": [{"path": best}], "note": "/kaggle/other"}))

    assert kaggle.merge_output(out, "aerotrack-w3", runs) == [rid]
    rec = json.loads((runs / rid / "run.json").read_text())
    local_best = f"{runs}/ultralytics/{rid}/weights/best.pt"
    assert rec["summary"]["weights"] == local_best and rec["artifacts"][0]["path"] == local_best
    assert rec["note"] == "/kaggle/other"
    assert rec["relocated"]["kernel"] == "aerotrack-w3"
    assert (runs / "ultralytics" / rid / "weights" / "best.pt").exists()
    assert (runs / "kaggle" / "aerotrack-w3" / "logs" / "utilisation.csv").exists()

    (out / "runs" / rid).mkdir(parents=True)
    try:
        kaggle.merge_output(out, "aerotrack-w3", runs)
    except SystemExit as e:
        assert "already exists" in str(e)
    else:
        raise AssertionError("an existing local run was overwritten")


def test_resume_push_points_the_job_at_the_checkpoint(monkeypatch):
    monkeypatch.setattr(kaggle, "bundle", lambda: ("a" * 40, _bundle_b64()))
    rid = "20261027-222023-visdrone-yolo26n-p2-960-93f3b8"
    folder = kaggle.push([CONFIGS_DIR / "train" / "smoke.yaml"], [], None, "aerotrack-test",
                         dry_run=True, resume=rid)
    src = (folder / "kernel.py").read_text()
    ns: dict = {}
    exec(src.split("\nROOT = ")[0], ns)
    [job] = ns["JOBS"]
    assert job["resume"] == rid
    assert f"resume_from={kaggle.RESUME_DIR}/{rid}" in job["overrides"]
    assert 'ROOT / "resume" / run_id' in src and kaggle.RESUME_DIR == "/tmp/aerotrack/resume"


def test_eval_job_names_its_input_runs_and_redirects_its_report(monkeypatch):
    monkeypatch.setattr(kaggle, "find_train_run", lambda ref: {"id": f"id-{ref}"})
    [job] = kaggle.make_jobs([CONFIGS_DIR / "eval" / "w6_keyframe.yaml"], [], None)
    assert job["kind"] == "evaluate" and job["dataset"] == "visdrone_mot"
    assert job["inputs"] == ["id-visdrone-yolo26n-640", "id-visdrone-yolo26n-1280"]
    assert f"report_dir={kaggle.KAGGLE_REPORTS}/w6-keyframe" in job["overrides"]


def test_merge_output_brings_reports_back_relocated(tmp_path):
    out, runs, reports = tmp_path / "out", tmp_path / "runs", tmp_path / "reports"
    (out / "reports" / "w8").mkdir(parents=True)
    w = f"{kaggle.KAGGLE_RUNS}/ultralytics/x/weights/best.pt"
    (out / "reports" / "w8" / "results.json").write_text(json.dumps({"results": [{"weights": w}]}))
    (out / "reports" / "w8" / "results.md").write_text("# w8\n")
    assert kaggle.merge_output(out, "aerotrack-w8", runs, reports) == ["reports/w8"]
    rec = json.loads((reports / "w8" / "results.json").read_text())
    assert rec["results"][0]["weights"] == f"{runs}/ultralytics/x/weights/best.pt"
