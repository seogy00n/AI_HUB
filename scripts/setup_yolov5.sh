#!/usr/bin/env bash
# YOLOv5 공식 저장소를 third_party/ 아래로 받고 의존성을 설치한다.
# third_party/는 .gitignore로 제외되어 있으므로 매번 이 스크립트로 준비한다.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p third_party
if [ ! -d third_party/yolov5 ]; then
  git clone https://github.com/ultralytics/yolov5 third_party/yolov5
fi

pip install -r third_party/yolov5/requirements.txt
echo "YOLOv5 ready at third_party/yolov5"
