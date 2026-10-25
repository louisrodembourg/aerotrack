#!/usr/bin/env bash
# One-command training on a fresh CUDA machine.
#
#   git clone <repo> && cd aerotrack
#   export WANDB_API_KEY=...            # results land in W&B; without it, runs stay offline
#   scripts/remote_train.sh configs/train/visdrone_yolo26s_640.yaml [--set train.epochs=50 ...]
#
# Modes:
#   default      Docker (needs docker + NVIDIA container toolkit, standard on GPU rentals)
#   --no-docker  installs uv and the locked environment on the host (e.g. Kaggle notebooks,
#                where custom images cannot be used)
#
# Data is downloaded and converted by the script itself (cached under $DATA_DIR), so the
# machine needs nothing but this repository.
set -euo pipefail

usage() { sed -n '2,15p' "$0"; exit 1; }
[[ $# -ge 1 ]] || usage

MODE=docker
if [[ "$1" == "--no-docker" ]]; then MODE=host; shift; fi
CONFIG=$1; shift
[[ -f "$CONFIG" ]] || { echo "config not found: $CONFIG" >&2; exit 1; }

REPO=$(cd "$(dirname "$0")/.." && pwd)
DATA_DIR=${DATA_DIR:-$REPO/data}
RUNS_DIR=${RUNS_DIR:-$REPO/runs}
WEIGHTS_DIR=${WEIGHTS_DIR:-$REPO/weights}
DATASET=$(grep -A2 '^data:' "$CONFIG" | awk '/dataset:/ {print $2}')
mkdir -p "$DATA_DIR" "$RUNS_DIR" "$WEIGHTS_DIR"

command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv \
    || echo "warning: nvidia-smi not found, training will not see a GPU" >&2
[[ -n "${WANDB_API_KEY:-}" ]] || echo "warning: WANDB_API_KEY unset, W&B runs stay offline in $RUNS_DIR/wandb" >&2

COMMIT=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)
DIRTY=$([[ -n "$(git -C "$REPO" status --porcelain --untracked-files=no 2>/dev/null)" ]] && echo true || echo false)
STEPS=("data download $DATASET" "data convert $DATASET" "train $CONFIG $*")

if [[ $MODE == docker ]]; then
    docker build -f "$REPO/docker/Dockerfile.cuda" --build-arg GIT_COMMIT="$COMMIT" \
        --build-arg GIT_DIRTY="$DIRTY" -t aerotrack:cuda "$REPO"
    for step in "${STEPS[@]}"; do
        # --ipc=host: dataloader workers exchange batches through shared memory.
        # shellcheck disable=SC2086
        docker run --rm --gpus all --ipc=host \
            -e WANDB_API_KEY="${WANDB_API_KEY:-}" -e WANDB_MODE="${WANDB_MODE:-}" \
            -e WANDB_PROJECT="${WANDB_PROJECT:-aerotrack}" -e WANDB_ENTITY="${WANDB_ENTITY:-}" \
            -v "$DATA_DIR:/data" -v "$RUNS_DIR:/runs" -v "$WEIGHTS_DIR:/weights" \
            aerotrack:cuda $step
    done
else
    command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH=$HOME/.local/bin:$PATH; }
    cd "$REPO" && uv sync --frozen --no-dev
    export AEROTRACK_DATA=$DATA_DIR AEROTRACK_RUNS=$RUNS_DIR AEROTRACK_WEIGHTS=$WEIGHTS_DIR
    for step in "${STEPS[@]}"; do
        # shellcheck disable=SC2086
        uv run --frozen aerotrack $step
    done
fi
echo "done: run records in $RUNS_DIR (run.json, weights under runs/ultralytics/<run-id>/)"
