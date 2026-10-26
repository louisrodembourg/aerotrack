# Decisions

One entry per non-trivial choice: what was chosen, what was rejected, why. Newest last.

## Tooling

**uv + universal lockfile, two platforms.** `tool.uv.environments` restricts resolution to
macOS arm64 and Linux x86_64. *Rejected:* conda (slow, no cross-platform lockfile), plain
pip + requirements (no lock across platforms). Jetson is out of the lock: JetPack ships
its own torch wheels; it will get its own environment at week 6 if it is chosen.

**torch from the cu126 index on Linux, PyPI on macOS.** PyPI's Linux wheels for torch 2.14
target CUDA 13, which needs an NVIDIA driver ≥ 580. Rented GPUs often run 550/570.
cu126 runs on any driver ≥ 525. *Rejected:* the default PyPI wheels (silent failure risk
on the remote box); cu128 (no torch 2.14.1 build published for it, checked on
download.pytorch.org).

**W&B for tracking, plus a local `runs/<id>/run.json`.** W&B mirrors remote runs without
copying files back. The local record is the source of truth for reports and the README,
so W&B being offline or down never loses a number. Without credentials, W&B runs offline
(`wandb sync` later). *Rejected:* MLflow (needs a reachable server or a manual `mlruns/`
sync from the remote machine).

**Ultralytics' own integrations disabled, settings repo-local.** `YOLO_CONFIG_DIR` points to
`.ultralytics/`, all logger integrations and analytics (`sync`) are off,
`YOLO_AUTOINSTALL=false`. Otherwise Ultralytics logs a second partial copy of each run,
and may `pip install` behind the lockfile's back.

**Docker for remote training, with a no-Docker fallback.** `scripts/remote_train.sh` builds
and runs `docker/Dockerfile.cuda` (base `nvidia/cuda:*-base`, since torch wheels bundle
CUDA/cuDNN). `--no-docker` runs the same steps on the host, for Kaggle where custom images
are impossible. The image has no `.git`: the commit is passed as a build arg and recorded
in each run. The base image comes from `nvcr.io` (NVIDIA's registry) rather than Docker
Hub: anonymous pulls, and no Docker Hub rate limit on shared GPU hosts.

## Data

**Own converters, not Ultralytics' `VisDrone.yaml` script.** That script *moves* the raw
images (`img.rename`) and drops ignored regions without masking them. It also sends
`others` (category 11) to class index 10, past the end of its 10-entry `names` (read in
ultralytics 8.4.187). Raw data must never be modified.

**VisDrone-DET from the Ultralytics GitHub mirror.** It is reliable, while the official
Google Drive links hit quota errors. Integrity: sha256 pinned in
`configs/data/visdrone_det.yaml`, and image counts checked against the VisDrone README
(6471 / 548 / 1610). *Not verified:* byte-equality with the Drive archives.

**Raw data read-only, processed images are symlinks.** Extracted files are `chmod a-w`.
`processed/*/images/` links to raw, so nothing is duplicated. Ultralytics does not resolve
symlinks when deriving label paths (checked in `data/base.py`). Our own code must not
resolve them either: one `.resolve()` in the subset writer sent Ultralytics looking for
labels in `data/raw/`, and it now uses `.absolute()`.

**Ignored regions: the official rule of each dataset, applied to both sides.**
- VisDrone (`dropObjects.m` in the MOT toolkit): a box is dropped when ≥ 50 % of its area
  lies in the *union* of category-0 and category-11 regions.
- UAVDT (`CalculateDetectionPR_*.m`): only detections strictly inside an ignore box are
  dropped.

The threshold is stored in the converted ground truth (`SplitGT.drop_coverage`).
*Rejected:* COCO `iscrowd` regions. They test overlap box by box (IoF), so a detection
straddling two adjacent ignore boxes survives, and VisDrone tiles crowds exactly that way.

**Ignored regions at training: masked on the fly.** `MaskedYOLODataset.load_image` paints
ignored pixels with the letterbox grey (114) before augmentation, keeping pixels inside
labeled boxes. *Rejected:* writing masked copies (double JPEG compression, 2 GB
duplicated); dropping the regions as Ultralytics does (ignored objects become
"background" negatives).

**Rows with score 0 on a real class → ignore region.** The README says they are "ignored in
evaluation". Measured: none exist in VisDrone-DET (all score-0 rows are category 0), so
this changes nothing today. The rule is in place for test-challenge-like files.

**Excluded classes (e.g. `van` for the COCO baseline) become ignore regions at
evaluation.** A COCO model calling a van a car is then neither rewarded nor penalised.
*Rejected:* dropping their ground truth only, which would count every such detection as a
false positive and bias the baseline against COCO models.

**COCO-baseline class map.** pedestrian + people → person, bicycle, car, truck, bus,
motor → motorcycle. Excluded: van (no COCO class, inconsistently car or truck in COCO),
tricycle, awning-tricycle. Mapping by class *name*, recorded in every report.

**TrackEval: one MOTChallenge benchmark per class.** `MotChallenge2DBox` only evaluates
`pedestrian` and rejects other class ids in tracker files (checked in its source), so
each class is written with class id 1 in its own folder. *Rejected:* the `Kitti2DBox` or
`BDD100K` loaders, which would need a format conversion of their own. The format is
validated by scoring the ground truth as a tracker: HOTA must be exactly 1.

**TrackEval numpy aliases restored at import.** Its last commit (12c8791) uses `np.int`,
`np.float` and `np.bool`, removed in numpy 1.24, in the MOTChallenge loader, HOTA and
Identity. These were aliases of builtins, so restoring them changes no behaviour.
*Rejected:* pinning numpy < 1.24 (incompatible with torch / Ultralytics), vendoring a
patched copy.

**VisDrone-MOT evaluated classes: pedestrian, car, van, truck, bus.** These are the
toolkit's five classes (README and `evaluateTrackA.m`). Other classes are kept out of the
benchmarks and counted in `stats.json`. Measured: 18 tracks on val and 229 on train change
class over their lifetime (e.g. car ↔ van). Each such track is split between two per-class
benchmarks. Ids cannot collide, since the benchmarks are separate, but the split costs
those tracks some IDF1 in a per-class evaluation. The count is reported in `stats.json`.

**UAVDT formats, read from the toolkit code (no README shipped).**
- `_gt_whole.txt` (category in column 9) is the DET ground truth.
- `_gt.txt` is the class-agnostic MOT ground truth. It drops objects marked out-of-view
  (column 7 ≠ 1 in gt_whole), so it is not row-identical to gt_whole (20 of 50 sequences
  differ). Classes are recovered by joining on (frame, id): 100 % of rows join.
- The class-agnostic `vehicle` benchmark is the official protocol; per-class benchmarks
  are ours.
- Deviation: the official MOT evaluation ignores ignore regions; we apply the UAVDT DET
  rule (fully inside) to both sides, which drops 614 of 340 906 test boxes.

**Sequence frame rate not written.** Neither dataset publishes it. `seqinfo.ini` omits
`frameRate` rather than inventing one; TrackEval only reads `seqLength`.

## Evaluation

**One `Detector` for accuracy and latency.** It runs preprocess → inference → postprocess,
with backends torch and onnxruntime, and TensorRT later behind the same interface.
*Rejected:* `model.val()` for accuracy plus a separate benchmark, which allows "accurate
with one preprocessing, fast with another".

**Fixed square letterbox (static batch-1 shape).** This is what an exported engine runs.
Ultralytics' `predict` pads to the minimum rectangle, so small mAP differences against
`yolo val` are expected. Agreement on detections is checked by a network test.

**COCO metrics with maxDets = 500, read from the precision array.** VisDrone images often
hold more than 100 objects, and the VisDrone protocol uses 500. `COCOeval.summarize()`
hard-codes the 100 column. Annotation ids start at 1, because pycocotools uses 0 as
"unmatched". COCO area ranges are kept for comparability; on VisDrone, "small" dominates.

**Latency: images pre-decoded, real images cycled, device sync after each stage.** Disk and
JPEG decoding are not part of the frame budget. NMS cost depends on the number of
candidates, so latency uses the deployment threshold (conf 0.25) while accuracy uses
0.001. End-to-end p95 is computed per iteration, not as a sum of per-stage p95.

**On MPS, postprocessing runs on the CPU.** Measured on M3 Pro: NMS / top-k on MPS costs
12–18 ms per image (overhead of many small kernels), against ~1.2 ms for copying the raw
output to the host and filtering there. The copy is billed to the postprocess stage. On
CUDA, NMS stays on the GPU.

**Tracking is not in the measured pipeline yet.** Until week 6, the reported FPS cover
pre + inference + post only, and every report says so.

## Baseline models

**YOLO26 as the reference family, YOLO11 as comparison, n and s for the baseline.**
- YOLO26 is NMS-free (`end2end: True` in `yolo26.yaml`) and has an official P2 config
  (`yolo26-p2.yaml`). The NMS stage of the budget is therefore an axis of analysis in
  itself; SAHI tile merging will still need NMS.
- YOLO11 keeps explicit NMS and has a more mature TensorRT INT8 track record.
- The n vs s choice is made in week 3, with the baseline numbers.

## Remote training (week 3)

**Kaggle, driven by the Kaggle CLI from the Mac.** Free, a fixed weekly GPU quota
(reported around 30 h by third-party sources; Kaggle's own pages could not be read
without a browser, so the figure is to be checked in the account), sessions of several
hours. The P100 is retired (`kaggle kernels push --help`, CLI 2.2.5): the free GPU is
2 × T4. *Rejected:* Colab (quota varies with demand, idle sessions are cut), paid
rentals (not needed until the measured throughput says so).

**The code travels inside the kernel script.** `push` embeds a base64 `git archive` of
HEAD (~75 KB) in the kernel. *Rejected:* a Kaggle dataset holding the code (one more
versioned object, and a race between dataset processing and the kernel start); cloning
from GitHub (needs a published repository or a token in Kaggle). Uncommitted changes are
refused, so the commit recorded in the run is exactly the code that ran.

**One job per T4, no DDP.** Ultralytics DDP re-imports the trainer class in a subprocess
from its module path (`generate_ddp_file`). Our trainer is built at run time to swap in
the masked dataset, and the tracked-run callbacks live in the parent process: both would
be lost. Two different runs in parallel (e.g. n and s) use the same session time instead.
The kernel samples GPU and CPU load every 30 s to show whether 2 jobs on Kaggle's vCPUs
are dataloader-bound.

**Kaggle CLI through `uvx`, pinned, not in the lockfile.** It only launches jobs from the
local machine; the training environment never needs it.

**Pulled runs are relocated, and the relocation is recorded.** Paths written on Kaggle
(`/kaggle/working/runs/...`) are rewritten to the local `runs/` and the run gets a
`relocated` field. Existing local runs are never overwritten.

## Evaluation protocol for fine-tuned models (week 3)

**Val selects, test-dev reports.** Ultralytics keeps `best.pt` by val fitness, so val
accuracy of a fine-tuned model is biased upwards. Reported numbers come from VisDrone-DET
test-dev (1610 images, labels public), which no choice ever looks at. The zero-shot
baseline is re-run on test-dev (`configs/eval/baseline_testdev.yaml`) so the comparison is
on the same images. *Rejected:* carving a held-out part out of train: its 6471 images share only 208
file-name prefixes, and a random split would put frames of one video on both sides.

The first field of a VisDrone-DET file name looks like a source-video id (the README does
not document it). Measured on those prefixes: test-dev shares none with train or val; val
shares 24 with train, covering 28 of its 548 images. Test-dev is therefore also the more
independent split.

**Fine-tuned vs zero-shot on identical classes.** A VisDrone fine-tune is also scored with
`visdrone_to_coco`: its predictions go through the same map as the ground truth
(pedestrian + people -> person, van dropped like the van ground truth). The gain from
fine-tuning is then measured on the classes the COCO model can name, next to the
10-class VisDrone score.

**Kaggle session measured (probe, 3 epochs, n and s in parallel, 2 workers each).**
Setup 2.3 min (uv sync 1.5, VisDrone-DET download + conversion 0.75). Training pass
~120 s/epoch for n and ~133 s for s; ~160 s/epoch with validation. The session is
CPU-bound: 4 vCPUs, load average 5.5, median GPU utilisation 31 % (n) and 70 % (s). The
100-epoch references therefore take ~4.5 h each, in parallel in one session.
*Not done:* `cache: ram` would remove JPEG decoding from the CPU budget, but Ultralytics
warns it breaks deterministic training (`data/base.py`), and runs already fit in one
session. To revisit if the week 4 sweeps are throughput-limited.

## Latency on the Mac (week 3)

**A throttled Mac is refused, and the power state is recorded.** The first week 3
evaluation ran overnight: the Mac slept and woke 217 times (`pmset -g log`), on battery
with Low Power Mode on. Its latency rows were wrong (YOLO26s faster than YOLO26n on
ONNX Runtime) and were discarded; accuracy does not depend on timing and was kept.
Measured in that state the next morning: YOLO26n zero-shot on MPS, 20.5 ms median
against 7.7 ms in the `baseline` report. Since then:
- every latency record carries `macos_power` (source, Low Power Mode), shown next to the
  device in reports, like the Jetson power mode;
- `aerotrack evaluate` refuses to time a Mac on battery or in Low Power Mode, before the
  accuracy pass, unless `latency.allow_throttled=true` (to measure that state on purpose);
- `make` wraps evaluations in `caffeinate -i`.

The baseline's power state was not recorded, so all Mac latency is re-measured in one
session under these rules. *Rejected:* detecting throttling from the timings themselves
(no reference to compare against on a new model).

**The ONNX export cache is keyed by the checkpoint's content.** It was keyed by file name,
and every fine-tune is `best.pt`: the first week 3 latency table timed, on ONNX Runtime,
the export of the 3-epoch YOLO26n probe for both 100-epoch references (YOLO26s showed
21.5 ms against 62.2 ms for the same architecture zero-shot, which exposed it). The key is
now `<stem>_<sha256[:12]>_<imgsz>_md<max_det>`. Release checkpoints were never affected
(unique names), so `baseline` stands; `w3-reference` was re-measured.
