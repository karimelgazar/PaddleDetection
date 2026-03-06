#!/usr/bin/env bash

# git clone -b main https://github.com/karimelgazar/PaddleDetection
apt remove python3-blinker
python -m pip install -U pip wheel blinker
# Keep pkg_resources available for PaddleDetection imports.
python -m pip install "setuptools==75.8.0"
# Known-good combo for PicoDet export in this workspace.
python -m pip install --force-reinstall "paddlepaddle-gpu==2.6.2" -i "https://www.paddlepaddle.org.cn/packages/stable/cu118/"
python -m pip install --force-reinstall "numpy<2"
python -m pip install -r requirements.txt
nvidia-smi
python check_setup.py

# Real install, not --dry-run.
# python -m pip install --no-cache-dir --force-reinstall \
#   "paddlepaddle-gpu==3.2.2" \
#   -i "https://www.paddlepaddle.org.cn/packages/stable/cu118/"


# python tools/yolo2coco.py --dataset_dir "$DATASET_DIR"

# THESE CLI Arguments below will override the ones in the config file, so we can test different settings without changing the config file.
# Keep LR linear to batch: new_lr = 0.08 * new_bs / 48
export CUDA_VISIBLE_DEVICES=0
mkdir -p logs
nohup python -u tools/train.py -c configs/picodet/picodet_s_416_coco_haramblur.yml -o use_gpu=True worker_num=3 TrainReader.use_shared_memory=False LearningRate.base_lr=0.10666  > logs/picodet_s_416_$(date +%F_%H-%M-%S).log 2>&1 &
tail -f logs/picodet_s_416_YYYY-MM-DD_HH-MM-SS.log


# python tools/train.py -c configs/picodet/picodet_s_416_coco_haramblur.yml -o use_gpu=True worker_num=0 TrainReader.batch_size=8 TrainReader.use_shared_memory=False LearningRate.base_lr=0.0133
python tools/train.py \
  -c configs/picodet/picodet_s_320_bird_detection.yml \
  --eval \
  -o use_gpu=True \
     TrainDataset.dataset_dir="$DATASET_DIR" \
     EvalDataset.dataset_dir="$DATASET_DIR" \
     TestDataset.dataset_dir="$DATASET_DIR"

# Resume training from a checkpoint:
nohup python -u tools/train.py -r /workspace/PaddleDetection/output/119.pdparams -c configs/picodet/picodet_s_416_coco_haramblur_big.yml -o use_gpu=True worker_num=16 TrainReader.use_shared_memory=False LearningRate.base_lr=0.426  > logs/picodet_s_416_$(date +%F_%H-%M-%S).log 2>&1 &

# Validation
python tools/eval.py -c configs/picodet/picodet_s_416_coco_haramblur_big.yml -o use_gpu=True worker_num=16 weights=/workspace/PaddleDetection/output/218.pdparams

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
cd PaddleDetection
python tools/export_model.py -c configs/picodet/picodet_s_416_coco_haramblur_big.yml \
              -o weights="/home/work/freelancer/PaddleDetection/output/haramblur/218.pdparams" \
              --output_dir="/home/work/freelancer/PaddleDetection/output/haramblur_exported_onnx"

# Convert exported Paddle model to ONNX (per configs/picodet/README_en.md).
python -m pip install -U "onnx==1.16.2" -i https://pypi.org/simple
python -m pip install -U "paddle2onnx==1.3.1" -i https://pypi.org/simple
paddle2onnx \
  --model_dir /home/work/freelancer/PaddleDetection/output/haramblur_exported_onnx/picodet_s_416_coco_haramblur_big/ \
  --model_filename model.pdmodel \
  --params_filename model.pdiparams \
  --opset_version 14 \
  --save_file /home/work/freelancer/PaddleDetection/output/haramblur_exported_onnx/picodet_s_416_coco_haramblur_big/model.onnx


