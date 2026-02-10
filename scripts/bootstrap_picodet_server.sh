#!/usr/bin/env bash

git clone -b main https://github.com/karimelgazar/PaddleDetection
apt remove python3-blinker
python -m pip install -U pip setuptools wheel blinker
python -m pip install -r requirements.txt

# Real install, not --dry-run.
# python -m pip install --no-cache-dir --force-reinstall \
#   "paddlepaddle-gpu==3.2.2" \
#   -i "https://www.paddlepaddle.org.cn/packages/stable/cu118/"

nvidia-smi
python check_setup.py

# python tools/yolo2coco.py --dataset_dir "$DATASET_DIR"

# THESE CLI Arguments below will override the ones in the config file, so we can test different settings without changing the config file.
# Keep LR linear to batch: new_lr = 0.08 * new_bs / 48
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/picodet/picodet_s_416_coco_haramblur.yml -o use_gpu=True worker_num=3 TrainReader.use_shared_memory=False LearningRate.base_lr=0.0133
# python tools/train.py -c configs/picodet/picodet_s_416_coco_haramblur.yml -o use_gpu=True worker_num=0 TrainReader.batch_size=8 TrainReader.use_shared_memory=False LearningRate.base_lr=0.0133
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
