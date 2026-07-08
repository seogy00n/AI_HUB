#!/usr/bin/env bash
# 학습된 가중치로 테스트 이미지에 대한 탐지 결과를 시각적으로 확인한다.
# 사용법: scripts/detect_yolov5.sh [weights_path] [source_dir]
set -euo pipefail
cd "$(dirname "$0")/.."

WEIGHTS=${1:-third_party/yolov5/runs/train/garbage_yolov5s/weights/best.pt}
SOURCE=${2:-data/yolo/images/test}

python third_party/yolov5/detect.py \
  --weights "$WEIGHTS" \
  --img 416 \
  --conf 0.4 \
  --source "$SOURCE"
