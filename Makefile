.PHONY: w3-eval w4-eval w4-p2-eval setup data data-det data-mot data-uavdt data-min viz check-data test lint smoke baseline readme docker-build help

RUN := uv run
CLI := $(RUN) aerotrack
# Latency runs keep a Mac awake: idle sleep throttled a whole night of measurements once.
KEEP_AWAKE := $(if $(filter Darwin,$(shell uname -s)),caffeinate -i,)
EVAL := $(KEEP_AWAKE) $(CLI) evaluate

help:  ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

setup:  ## install the locked environment (macOS arm64 or Linux x86_64)
	uv sync --frozen

data: data-det data-uavdt data-mot viz check-data  ## download + convert + visualise + check every dataset

data-min: data-det  ## only what smoke and baseline need (VisDrone-DET, ~2 GB)

data-det:
	$(CLI) data download visdrone_det
	$(CLI) data convert visdrone_det

data-mot:
	$(CLI) data download visdrone_mot
	$(CLI) data convert visdrone_mot

data-uavdt:
	$(CLI) data download uavdt
	$(CLI) data convert uavdt

viz:  ## draw converted labels on random frames -> reports/viz/
	$(CLI) data viz visdrone_det visdrone_mot uavdt

check-data:  ## converted MOT ground truth, scored by TrackEval as a tracker, must give HOTA = 1
	$(CLI) data check-trackeval visdrone_mot --split val
	$(CLI) data check-trackeval uavdt --split test

test:  ## offline unit tests (add --run-network for the Ultralytics consistency test)
	$(RUN) pytest -q

lint:
	$(RUN) ruff check src tests

smoke:  ## ~1 min: short training + evaluation on 100 images, on the local device
	$(CLI) train configs/train/smoke.yaml

baseline:  ## COCO-pretrained YOLO on VisDrone-DET val, inference only
	$(EVAL) configs/eval/baseline.yaml
	$(CLI) readme-table

w3-eval:  ## week 3: fine-tuned references on test-dev (pull them from Kaggle first), then README
	$(EVAL) configs/eval/w3_reference.yaml
	$(EVAL) configs/eval/w3_vs_zero_shot.yaml
	$(CLI) readme-table

w4-eval:  ## week 4: YOLO26n resolution sweep on test-dev (pull from Kaggle first), then README
	$(EVAL) configs/eval/w4_resolution.yaml
	$(CLI) readme-table

w4-p2-eval:  ## week 4: YOLO26n P2 head vs plain at 640 and 960 on test-dev, then README
	$(EVAL) configs/eval/w4_p2.yaml
	$(CLI) readme-table

readme:  ## regenerate README results from tracked reports
	$(CLI) readme-table

docker-build:  ## CUDA training image (linux/amd64; builds under emulation on Apple Silicon)
	docker buildx build --platform linux/amd64 -f docker/Dockerfile.cuda -t aerotrack:cuda --load .
