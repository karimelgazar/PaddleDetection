#!/usr/bin/env bash
set -euo pipefail

# =========================
# Customize these variables
# =========================
REPO_URL="${REPO_URL:-https://github.com/<YOUR_USER_OR_ORG>/PaddleDetection.git}"
REPO_REF="${REPO_REF:-<YOUR_BRANCH_OR_COMMIT>}"
WORKDIR="${WORKDIR:-$HOME/work}"
ENV_NAME="${ENV_NAME:-rag_saas}"
GPU_ID="${GPU_ID:-0}"

DATASET_DIR="${DATASET_DIR:-$HOME/datasets/BirdDetection.v2-initial-dataset.yolo26}"
SAMPLE_IMAGE="${SAMPLE_IMAGE:-$DATASET_DIR/test/images/sample.jpg}"

mkdir -p "$WORKDIR"
cd "$WORKDIR"

if [ ! -d PaddleDetection ]; then
  git clone "$REPO_URL" PaddleDetection
fi

cd PaddleDetection
git fetch --all --tags
git checkout "$REPO_REF"

source ~/.zshrc

if ! conda run -n "$ENV_NAME" python -V >/dev/null 2>&1; then
  conda create -y -n "$ENV_NAME" python=3.12
fi
conda activate "$ENV_NAME"

python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt

# Real install, not --dry-run.
python -m pip install --no-cache-dir --force-reinstall \
  "paddlepaddle-gpu==3.2.2" \
  -i "https://www.paddlepaddle.org.cn/packages/stable/cu118/"

nvidia-smi
export CUDA_VISIBLE_DEVICES="$GPU_ID"

python check_setup.py

python tools/yolo2coco.py --dataset_dir "$DATASET_DIR"

python tools/train.py \
  -c configs/picodet/picodet_s_320_bird_detection.yml \
  --eval \
  -o use_gpu=True \
     TrainDataset.dataset_dir="$DATASET_DIR" \
     EvalDataset.dataset_dir="$DATASET_DIR" \
     TestDataset.dataset_dir="$DATASET_DIR"

BEST_WEIGHTS="output/picodet_s_320_bird_detection/best_model/model.pdparams"
[ -f "$BEST_WEIGHTS" ] || { echo "Best weights not found: $BEST_WEIGHTS"; exit 1; }

python tools/infer.py \
  -c configs/picodet/picodet_s_320_bird_detection.yml \
  -o use_gpu=True \
     weights="$BEST_WEIGHTS" \
     TestDataset.dataset_dir="$DATASET_DIR" \
  --output_dir output_inference \
  --infer_img "$SAMPLE_IMAGE"

python tools/export_model.py \
  -c configs/picodet/picodet_s_320_bird_detection.yml \
  -o weights="$BEST_WEIGHTS" \
     TestDataset.dataset_dir="$DATASET_DIR" \
  --output_dir output_export

echo "Done"
echo "Best weights: $(pwd)/$BEST_WEIGHTS"
echo "Inference outputs: $(pwd)/output_inference"
echo "Exported model: $(pwd)/output_export"
