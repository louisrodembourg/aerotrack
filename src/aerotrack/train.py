"""Fine-tuning with Ultralytics, wrapped so that every run is tracked and every image the
network sees has its ignored regions masked.

Masking happens on the fly in the dataset, not in a re-encoded copy of the images:
no second JPEG compression, no duplicated dataset on disk, and raw data stays untouched.
"""

from __future__ import annotations

import json
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from aerotrack.data.classes import get_class_map
from aerotrack.data.copy_paste import build_bank, paste, rare_classes
from aerotrack.data.gt import SplitGT
from aerotrack.data.ignore import paint_ignored
from aerotrack.eval.detection import evaluate_detection
from aerotrack.experiments import start_run
from aerotrack.inference import Detector
from aerotrack.paths import PROCESSED_DIR, RUNS_DIR
from aerotrack.runtime import resolve_device, seed_everything
from aerotrack.ultra import configure, load_yolo, resolve_weights


def ignore_path(image_path: str) -> Path:
    head, sep, tail = image_path.rpartition("/images/")
    if not sep:
        raise ValueError(f"{image_path}: expected an .../images/<split>/... layout")
    return Path(head) / "ignore" / Path(tail).with_suffix(".txt")


def _masked_dataset_class():
    from ultralytics.data import YOLODataset  # noqa: PLC0415

    class MaskedYOLODataset(YOLODataset):
        """YOLODataset whose images have their ignored regions painted grey before any
        augmentation (mosaic, crops...), so ignored content can never be cut into a
        training sample as unlabeled background."""

        def load_ignore(self) -> None:
            self.ignore_boxes = []
            for f in self.im_files:
                p = ignore_path(f)
                text = p.read_text().strip() if p.exists() else ""
                self.ignore_boxes.append(np.loadtxt(text.splitlines(), ndmin=2) if text
                                         else np.zeros((0, 4)))

        def load_image(self, i, rect_mode=True, resize_short=False):
            im, (h0, w0), (h, w) = super().load_image(i, rect_mode, resize_short)
            ign = self.ignore_boxes[i]
            if len(ign):
                scale = np.array([w / w0, h / h0, w / w0, h / h0])
                b = self.labels[i]["bboxes"]  # normalised cx, cy, w, h
                keep = np.column_stack([(b[:, 0] - b[:, 2] / 2) * w, (b[:, 1] - b[:, 3] / 2) * h,
                                        b[:, 2] * w, b[:, 3] * h]) if len(b) else np.zeros((0, 4))
                # In place, so a cached image is masked too; painting is idempotent.
                paint_ignored(im, ign * scale, keep)
            return im, (h0, w0), (h, w)

        def enable_copy_paste(self, cp: dict) -> dict:
            """Build the rare-object bank (aerotrack.data.copy_paste) for this dataset."""
            counts = Counter(int(c) for lb in self.labels for c in lb["cls"].reshape(-1))
            classes = cp.get("classes") or rare_classes(counts, cp["rare_share"])
            self.cp = cp
            self.cp_bank = build_bank(self.labels, classes, cp["min_side"], cp["max_side"])
            self.cp_index = {f: i for i, f in enumerate(self.im_files)}
            return {"classes": classes, "bank_size": len(self.cp_bank),
                    "bank_per_class": dict(Counter(self.cp_bank.cls.tolist()))}

        def update_labels_info(self, label):
            if getattr(self, "cp_bank", None) is not None and np.random.random() < self.cp["p"]:
                label = self._paste(label)
            return super().update_labels_info(label)

        def _paste(self, label):
            # A copy: the image may be the one Ultralytics buffers for mosaic, and pasting
            # into it would accumulate objects across reads.
            img = label["img"].copy()
            h, w = img.shape[:2]
            (h0, w0) = label["ori_shape"]
            b = label["bboxes"]
            boxes = np.column_stack([(b[:, 0] - b[:, 2] / 2) * w, (b[:, 1] - b[:, 3] / 2) * h,
                                     (b[:, 0] + b[:, 2] / 2) * w, (b[:, 1] + b[:, 3] / 2) * h]
                                    ) if len(b) else np.zeros((0, 4))
            ign = self.ignore_boxes[self.cp_index[label["im_file"]]] * np.array(
                [w / w0, h / h0, w / w0, h / h0])
            ign = np.column_stack([ign[:, :2], ign[:, :2] + ign[:, 2:4]]) if len(ign) \
                else np.zeros((0, 4))
            n = int(np.random.randint(1, self.cp["max_paste"] + 1))
            rng = np.random.default_rng(np.random.randint(2**31))
            img, new, cls = paste(img, boxes, ign, self.cp_bank, w / w0, n, rng)
            if len(new):
                xywh = np.column_stack([(new[:, 0] + new[:, 2]) / 2 / w, (new[:, 1] + new[:, 3]) / 2 / h,
                                        (new[:, 2] - new[:, 0]) / w, (new[:, 3] - new[:, 1]) / h])
                label["bboxes"] = np.concatenate([b, xywh]).astype(np.float32)
                label["cls"] = np.concatenate([label["cls"], cls.reshape(-1, 1)]).astype(
                    label["cls"].dtype)
            label["img"] = img
            return label

    return MaskedYOLODataset


def _masked_trainer_class(copy_paste: dict | None = None, report: dict | None = None):
    from ultralytics.models.yolo.detect import DetectionTrainer  # noqa: PLC0415

    masked = _masked_dataset_class()

    class MaskedDetectionTrainer(DetectionTrainer):
        def build_dataset(self, img_path, mode="train", batch=None):
            # Ultralytics picks the dataset class inside build_yolo_dataset; re-classing the
            # instance keeps its construction logic (augmentations, rect, caching) intact
            # across Ultralytics versions instead of copying it here.
            ds = super().build_dataset(img_path, mode, batch)
            ds.__class__ = masked
            ds.load_ignore()
            if copy_paste and mode == "train":  # never on validation images
                info = ds.enable_copy_paste(copy_paste)
                if report is not None:
                    report.update(info)
            return ds

    return MaskedDetectionTrainer


def write_subset(dataset: str, name: str, limits: dict, seed: int) -> Path:
    """Ultralytics dataset yaml over a seeded subset of each split (image list files)."""
    root = PROCESSED_DIR / dataset
    base = yaml.safe_load((root / "dataset.yaml").read_text())
    out = root / "subsets"
    out.mkdir(parents=True, exist_ok=True)
    cfg = {"path": str(root.resolve()), "names": base["names"]}
    for split, n in limits.items():
        gt = SplitGT.load(root / "gt" / f"{split}.json").subset(n, seed=seed)
        # absolute(), not resolve(): resolving would follow the symlink into data/raw, where
        # Ultralytics would then look for labels (and write its cache).
        files = [str((root / "images" / split / Path(im.file).name).absolute()) for im in gt.images]
        lst = out / f"{name}_{split}.txt"
        lst.write_text("\n".join(files) + "\n")
        cfg[split] = str(lst)
    path = out / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def run_training(cfg: dict) -> dict:
    configure()
    seed = cfg.get("seed", 0)
    seed_everything(seed)
    device = resolve_device(cfg.get("device", "auto"))
    data_cfg, tr = cfg["data"], cfg["train"]
    dataset = data_cfg["dataset"]
    limits = {"train": data_cfg.get("train_limit"), "val": data_cfg.get("val_limit")}
    if any(limits.values()):
        data_yaml = write_subset(dataset, cfg["name"], limits, seed)
    else:
        data_yaml = PROCESSED_DIR / dataset / "dataset.yaml"
    n_train = len(SplitGT.load(PROCESSED_DIR / dataset / "gt" / "train.json")
                  .subset(limits["train"], seed=seed).images)

    with start_run(cfg["name"], "train", cfg, seed=seed, device=device, tags=("train",)) as run:
        save_dir = RUNS_DIR / "ultralytics" / run.id
        if cfg.get("resume_from"):
            # Continue an interrupted run (Kaggle session deadline) from its last epoch, with
            # its optimizer, schedule and best fitness. The weights are copied into this
            # run's directory so the interrupted run stays untouched.
            src = Path(cfg["resume_from"])
            (save_dir / "weights").mkdir(parents=True, exist_ok=True)
            for f in ("last.pt", "best.pt"):
                if (src / f).exists():
                    shutil.copy2(src / f, save_dir / "weights" / f)
            resume = {"resume": str(save_dir / "weights" / "last.pt"), "save_dir": str(save_dir),
                      "workers": tr.get("workers", 8)}
            model = load_yolo(resume["resume"])
        else:
            resume = None
            model = load_yolo(cfg["model"])
            if cfg.get("pretrained"):
                # An architecture built from .yaml starts random: transfer every tensor of the
                # release weights whose name and shape match (backbone, most of the neck).
                model.load(resolve_weights(cfg["pretrained"]))
        timing = {"train_seconds": []}

        def on_train_epoch_start(trainer):
            timing["t0"] = time.perf_counter()

        def on_train_epoch_end(trainer):
            # Training pass only (validation excluded): this is the throughput figure used
            # to decide whether a full run fits on this machine.
            timing["train_seconds"].append(time.perf_counter() - timing["t0"])

        def on_fit_epoch_end(trainer):
            metrics = {**trainer.label_loss_items(trainer.tloss, prefix="train"),
                       **{k: float(v) for k, v in (trainer.metrics or {}).items()},
                       "train_seconds": timing["train_seconds"][-1],
                       "lr": float(trainer.optimizer.param_groups[0]["lr"])}
            run.log({k: float(v) for k, v in metrics.items()}, step=trainer.epoch + 1)

        model.add_callback("on_train_epoch_start", on_train_epoch_start)
        model.add_callback("on_train_epoch_end", on_train_epoch_end)
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

        cp, cp_info = cfg.get("copy_paste"), {}
        t_start = time.perf_counter()
        if resume:
            # Ultralytics takes every other argument from the checkpoint; only these may change.
            model.train(data=str(data_yaml), trainer=_masked_trainer_class(cp, cp_info),
                        device=device, **resume)
        else:
            model.train(
                data=str(data_yaml), trainer=_masked_trainer_class(cp, cp_info), device=device,
                seed=seed,
                project=str(RUNS_DIR / "ultralytics"), name=run.id, exist_ok=True,
                verbose=False, **tr,
            )
        wall = time.perf_counter() - t_start
        save_dir = Path(model.trainer.save_dir)
        best = save_dir / "weights" / "best.pt"

        last_epoch = timing["train_seconds"][-1]
        throughput = n_train / last_epoch
        full_train = len(SplitGT.load(PROCESSED_DIR / dataset / "gt" / "train.json").images)
        summary = {
            "weights": str(best),
            "wall_seconds": wall,  # this run only: a resumed run excludes the epochs before
            "resumed_from": cfg.get("resume_from"),
            "copy_paste": cp_info or None,  # rare classes used and the bank actually built
            "train_images": n_train,
            "epochs": tr["epochs"],
            # Ultralytics halves the batch on a first-epoch CUDA OOM: record what actually
            # ran. Gradients accumulate to the nominal batch (nbs=64) either way.
            "batch_effective": model.trainer.batch_size,
            "accumulate": model.trainer.accumulate,
            "nominal_batch": model.trainer.args.nbs,
            "train_seconds_per_epoch": timing["train_seconds"],
            "train_images_per_second_last_epoch": throughput,
            "extrapolated_full_epoch_minutes": full_train / throughput / 60,
            "extrapolation_note": (f"{full_train} training images at the last epoch's rate, "
                                   f"imgsz={tr.get('imgsz')}, batch={tr.get('batch')}, "
                                   "validation time excluded"),
        }

        ev = cfg.get("evaluate")
        if ev:
            gt = SplitGT.load(PROCESSED_DIR / dataset / "gt" / f"{ev.get('split', 'val')}.json")
            gt = gt.subset(ev.get("limit"), seed=seed)
            det = Detector(best, backend="torch", device=device, imgsz=ev["imgsz"],
                           conf=ev["conf"], iou=ev["iou"], max_det=ev["max_det"])
            acc = evaluate_detection(gt, get_class_map(ev["class_map"]), det, det.names,
                                     ev["max_det"])
            summary["accuracy"] = acc
            print(f"[{cfg['name']}] mAP50-95={acc['mAP50-95']:.4f} mAP50={acc['mAP50']:.4f} "
                  f"on {acc['n_images']} images")

        run.summary(summary)
        (run.dir / "result.json").write_text(json.dumps(summary, indent=2))
        run.artifact(best, kind="model")
        run.artifact(save_dir / "results.csv", kind="training-log")
        print(f"[{cfg['name']}] training throughput {throughput:.1f} img/s on {device}; "
              f"a full epoch ({full_train} images) ~ {summary['extrapolated_full_epoch_minutes']:.1f} min")
        return summary
