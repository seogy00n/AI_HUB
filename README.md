# ♻️ 웹캠 기반 실시간 재활용품 분류

웹캠으로 물체를 비추면 실시간으로 종류(`cardboard`, `paper`, `metal`, `plastic`, `glass`, `trash`)를 탐지·분류해 화면에 바운딩 박스와 신뢰도를 표시하는 프로젝트.

대구 AI_HUB Living Lab+ (2024.07~08)에서 시작한 프로젝트를 실제 저장소 구조로 재구성한 버전입니다. 원본 배경/트러블슈팅/회고는 Notion 문서를 참고하세요.

## 파이프라인

```
데이터 수집 (Kaggle)
   ↓
라벨링 / 정제 (CVAT, YOLO 포맷)
   ↓
1단계 분류 실험: ResNet50 전이학습 (PyTorch)
   ↓
2단계 최종 채택: YOLOv5s 객체 탐지 (freeze fine-tuning)
   ↓
TensorFlow.js 변환
   ↓
웹캠 실시간 추론 (브라우저, HTML/CSS/JS)
```

## 저장소 구조

```
configs/data.yaml     # YOLOv5 데이터셋 설정
data/                 # 학습 데이터 (git 미포함, data/README.md 참고)
src/
  data/dataset.py      # 분류 데이터셋 로딩 / DataLoader 유틸
  models/resnet_classifier.py  # ResNet50 전이학습 모델
  train_classifier.py  # 분류기 학습 CLI
  evaluate_classifier.py  # 테스트셋 평가 + confusion matrix
  predict.py            # 단일 이미지 추론
scripts/
  setup_yolov5.sh       # YOLOv5 저장소 클론 + 의존성 설치
  train_yolov5.sh        # YOLOv5 fine-tuning
  detect_yolov5.sh        # 테스트 이미지 탐지 결과 확인
  export_tfjs.sh           # PyTorch → TensorFlow.js 변환
web/
  index.html, style.css, script.js  # 웹캠 실시간 추론 데모
```

## 시작하기

### 0. 환경 준비

```bash
pip install -r requirements.txt
```

### 1. 데이터 준비

`data/README.md`의 폴더 구조에 맞게 이미지를 넣습니다. (분류용 `data/classification/`, 탐지용 `data/yolo/`)

### 2. (선택) ResNet50 분류 실험

```bash
python -m src.train_classifier --data-dir data/classification --epochs 6
python -m src.evaluate_classifier --checkpoint outputs/classifier/model.bin
```

### 3. YOLOv5 탐지 모델 학습 (최종 채택 모델)

```bash
bash scripts/setup_yolov5.sh
bash scripts/train_yolov5.sh
bash scripts/detect_yolov5.sh   # 결과 시각 확인
```

### 4. 웹 배포용 변환 + 데모 실행

```bash
bash scripts/export_tfjs.sh
python -m http.server 8000 --directory web
# 브라우저에서 http://localhost:8000 접속, 웹캠 권한 허용
```

## 이번 재구성에서 바뀐 점

- 기존에는 코드가 Colab 노트북 셀과 Notion 텍스트로만 존재 → CLI로 실행 가능한 스크립트/모듈로 정리
- `ResNetClassifier.forward()`에서 다중분류에 맞지 않는 `sigmoid` 적용 제거 (raw logits + `cross_entropy` 사용)
- YOLOv5 학습 시 불필요했던 `model.yaml` 커스텀 config 복사 단계를 제거하고 `--data`만으로 클래스 수를 인식하도록 단순화
- YOLO 데이터셋을 표준 `images/`, `labels/` 분리 구조로 정리 (기존에는 클래스 폴더 안에 이미지·라벨이 섞여 있었음)
- 문서화되어 있지 않던 3단계(웹 프론트엔드)를 TensorFlow.js 기반 실시간 추론 데모로 신규 구현 (NMS, confidence threshold 슬라이더 포함)

## 알려진 한계 (다음 개선 후보)

- 학습 데이터 약 2,527장으로 정확도 확보에 부족 — 데이터 증강/추가 수집 필요
- k-fold 교차 검증 등 다른 파인튜닝 기법 미시도
- 배치 사이즈 / epoch / freeze 레이어 수에 대한 체계적 탐색 없음
- YOLOv5(2020)는 더 최신 버전(YOLOv8/11 등)으로 마이그레이션 여지가 있음

## 참고 문서 (Notion)

- [재활용품 실시간 분류 시스템 — 연구 보고서](https://app.notion.com/p/3967d7f5d24681129a40ce237bdad5cf)
- [AI_HUB 프로젝트 설명](https://app.notion.com/p/5daf4a41f2834e07abf3b2943af08855)
- [대구 AI_HUB Living Lab+ (24.07~24.08)](https://app.notion.com/p/36b7d7f5d2468064826fe9463ca4e975)
