#!/usr/bin/env bash
# 재활용품 데이터로 YOLOv5s를 fine-tuning한다. 먼저 setup_yolov5.sh를 실행해둘 것.
set -euo pipefail
cd "$(dirname "$0")/.."

python third_party/yolov5/train.py \
  --img 416 \
  --batch 16 \
  --epochs 32 \
  --freeze 10 \
  --data configs/data.yaml \
  --weights yolov5s.pt \
  --name garbage_yolov5s \
  --cache "$@"
