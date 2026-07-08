#!/usr/bin/env bash
# 학습된 YOLOv5 가중치를 TensorFlow.js로 변환하고 web/model/에 배치한다.
# 사용법: scripts/export_tfjs.sh [weights_path]
set -euo pipefail
cd "$(dirname "$0")/.."

WEIGHTS=${1:-third_party/yolov5/runs/train/garbage_yolov5s/weights/best.pt}

pip install "tensorflowjs==4.19.0" "tensorflow==2.13.1"
python third_party/yolov5/export.py --weights "$WEIGHTS" --include tfjs --img 416

WEB_MODEL_DIR="$(dirname "$WEIGHTS")/$(basename "$WEIGHTS" .pt)_web_model"
mkdir -p web/model
cp "$WEB_MODEL_DIR"/model.json web/model/
cp "$WEB_MODEL_DIR"/*.bin web/model/

echo "TFJS model copied to web/model/"
