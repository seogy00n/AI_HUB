# 📄 웹캠 기반 재활용품 실시간 분류 시스템 — 연구 보고서 (v2, 코드 기준)

> 2024.07~08 대구 AI_HUB Living Lab+에서 수행한 프로젝트를, 2026.07 저장소 재구성 이후의 **실제 코드**를 기준으로 다시 정리한 보고서입니다. 원본 프로젝트의 배경/회고는 Notion의 1차 연구 보고서를, 이번 재구성 작업 이력은 개발 재구성 로그를 참고하세요.

## 1. 연구 배경 및 목적

### 1.1 배경
- Living Lab+ 교육과정의 실생활 문제 기반 프로젝트 요구에 따라, 일상에서 친숙한 **비전 데이터 기반 객체 인식**을 주제로 선정
- **환경 문제 해결**을 목표로 한 AI 응용 프로젝트

### 1.2 문제 정의
- 웹캠 입력에서 재활용품을 **실시간으로 탐지·분류**하여 사용자에게 직관적인 피드백 제공

### 1.3 목표
- (정성) 카메라 앞에 물체를 보여주면 즉시 종류를 알 수 있는 웹 페이지
- (정량) 실시간 추론이 가능한 수준의 속도·정확도 확보

## 2. 시스템 아키텍처

```
데이터 수집 (Kaggle)
   ↓
라벨링 / 정제 (CVAT, YOLO 포맷: images/ + labels/ 분리)
   ↓
[실험] 1단계 분류: ResNet50 전이학습 (src/train_classifier.py)
   ↓
[채택] 2단계 탐지: YOLOv5s fine-tuning (scripts/train_yolov5.sh)
   ↓
TensorFlow.js 변환 (scripts/export_tfjs.sh)
   ↓
웹캠 실시간 추론 (web/, 브라우저에서 tf.js로 직접 실행)
```

- **분류 클래스 (6종)**: `cardboard`, `paper`, `metal`, `plastic`, `glass`, `trash`
- **운영 환경**: 로컬/서버에서 학습(PyTorch, GPU 권장) → 정적 웹 페이지에서 추론(서버리스, 클라이언트 GPU/CPU 사용)

## 3. 저장소 구조와 각 파일의 역할

| 경로 | 역할 |
|---|---|
| `configs/data.yaml` | YOLOv5 데이터셋 정의 (`path`, `train/val/test`, `nc: 6`, `names`) |
| `data/README.md` | 학습 데이터 배치 규칙 (분류용 `data/classification/`, 탐지용 `data/yolo/images,labels`) — 데이터 자체는 git 미포함 |
| `src/data/dataset.py` | `ImageFolder` 기반 분류 데이터셋 로딩, train/val/test 분할, `DeviceDataLoader` (GPU 전송 래퍼) |
| `src/models/resnet_classifier.py` | ImageNet 사전학습 ResNet50 전이학습 모델, 공통 학습/검증 스텝 정의 |
| `src/train_classifier.py` | 분류기 학습 CLI (`python -m src.train_classifier`), 가중치·클래스 목록·정확도 그래프 저장 |
| `src/evaluate_classifier.py` | 테스트셋 정확도 및 confusion matrix 산출 |
| `src/predict.py` | 단일 이미지 추론 CLI |
| `scripts/setup_yolov5.sh` | `ultralytics/yolov5` 클론 + 의존성 설치 (`third_party/`, git 미포함) |
| `scripts/train_yolov5.sh` | YOLOv5s fine-tuning 실행 (freeze 10, img 416, batch 16, epochs 32) |
| `scripts/detect_yolov5.sh` | 학습된 가중치로 테스트 이미지 탐지 시각 검증 |
| `scripts/export_tfjs.sh` | PyTorch(.pt) → TensorFlow.js(model.json + shard) 변환, `web/model/`로 배치 |
| `web/index.html`, `style.css`, `script.js` | 브라우저 웹캠 스트림을 받아 tf.js 모델로 실시간 탐지, NMS 후처리, confidence 슬라이더 제공 |

## 4. 데이터셋

- **출처**: Kaggle 재활용품 이미지 데이터셋
- **규모(원본 기준)**: 총 2,527장 — train 1,593 / val 176 / test 758 (랜덤 시드 42 고정)
- **전처리**: 256×256 리사이즈 후 텐서 변환(분류), CVAT로 YOLO 포맷 바운딩 박스 어노테이션(탐지)
- **현재 상태**: 이번 재구성 시점에는 로컬에 데이터가 없어 코드/파이프라인만 구축한 상태이며, 데이터 반입 후 `data/README.md` 구조에 맞춰 재학습 예정

## 5. 모델 1 — ResNet50 분류기 (`src/models/resnet_classifier.py`)

- ImageNet 사전학습 `resnet50`을 불러와 마지막 FC 레이어를 클래스 수(6)에 맞게 교체하는 전형적인 전이학습 구조
- 옵티마이저 Adam, 학습률 5.5e-5, 기본 6 epoch (`src/train_classifier.py` CLI 인자로 조정 가능)
- 손실 함수: `cross_entropy` (다중 클래스 분류 표준)

### 이번 재구성에서 수정한 버그
기존 Colab 코드는 `forward()`에서 `torch.sigmoid(self.network(xb))`를 출력했습니다. `sigmoid`는 각 클래스를 독립적인 이진 문제로 다루는 **multi-label** 설정에 적합하며, `cross_entropy`와 함께 쓰는 **multi-class** 분류에서는 정석이 아닙니다(정석은 raw logits을 그대로 `cross_entropy`에 전달). 재구성된 `ResNetClassifier.forward()`는 activation 없이 raw logits을 반환하도록 수정했습니다.

## 6. 모델 2 — YOLOv5s 탐지기 (채택 모델)

- 실시간성이 요구되어 1-stage detector인 **YOLOv5s** 채택
- 학습 명령 (`scripts/train_yolov5.sh`):
  ```bash
  python third_party/yolov5/train.py \
    --img 416 --batch 16 --epochs 32 --freeze 10 \
    --data configs/data.yaml --weights yolov5s.pt \
    --name garbage_yolov5s --cache
  ```
- 앞쪽 레이어 10개를 freeze하고 뒷단 위주로 fine-tuning
- `configs/data.yaml`에서 `nc: 6`, `names`로 클래스를 정의하므로 별도의 커스텀 `model.yaml` 복사 단계 없이 바로 학습 가능(기존 대비 단순화)
- `scripts/detect_yolov5.sh`로 `--conf 0.4` 기준 탐지 결과를 시각적으로 검증

### YOLO 데이터셋 레이아웃 변경
기존에는 클래스별 폴더 안에 이미지와 라벨(.txt)이 함께 있는 구조였습니다(우연히 YOLOv5의 기본 라벨 경로 치환 규칙이 무력화되어 동작). 재구성된 구조는 표준 관례인 `images/{train,val,test}` + `labels/{train,val,test}` 분리 방식을 따릅니다.

## 7. 웹 배포 — TensorFlow.js 변환 및 프론트엔드

### 7.1 변환 (`scripts/export_tfjs.sh`)
```bash
python third_party/yolov5/export.py --weights <best.pt> --include tfjs --img 416
```
`model.json` + `group1-shardXofY.bin` 파일을 `web/model/`로 복사.

### 7.2 프론트엔드 (`web/script.js`)
- `tf.loadGraphModel('./model/model.json')`로 모델 로드
- `navigator.mediaDevices.getUserMedia`로 웹캠 스트림 획득
- 매 프레임: 416×416 리사이즈 → 정규화(0~1) → `model.executeAsync()`
- YOLOv5 출력(`[cx, cy, w, h, objectness, class0..class5]`)을 디코딩하여 `objectness × classScore`로 신뢰도 계산
- `tf.image.nonMaxSuppressionAsync`로 NMS 수행(IoU 0.45, 최대 20개 박스)
- `<canvas>`에 바운딩 박스 + 클래스명 + 신뢰도(%) 오버레이, UI 슬라이더로 confidence threshold(기본 0.4) 실시간 조정

이 부분은 원본 프로젝트에서 문서화되어 있지 않던 단계였으며, 이번 재구성에서 신규로 구현했습니다.

## 8. 재현 방법 (Quickstart)

```bash
pip install -r requirements.txt

# 1) 데이터 배치 (data/README.md 참고)
# 2) (선택) 분류 실험
python -m src.train_classifier --data-dir data/classification --epochs 6
python -m src.evaluate_classifier --checkpoint outputs/classifier/model.bin

# 3) 탐지 모델 학습 (채택 모델)
bash scripts/setup_yolov5.sh
bash scripts/train_yolov5.sh
bash scripts/detect_yolov5.sh

# 4) 웹 배포
bash scripts/export_tfjs.sh
python -m http.server 8000 --directory web
```

## 9. 트러블슈팅 (원본 프로젝트 기준)

- **과적합**: early stopping, epoch 조절로 대응 (재구성된 `src/train_classifier.py`는 `--patience` 옵션으로 실제 early stopping 로직을 제공)
- **랜덤 시드 고정(42)**: 매 실행마다 동일한 데이터 분할을 보장해 다른 하이퍼파라미터의 영향을 통제

## 10. 한계 및 향후 계획

- 학습 데이터 약 2,527장은 정확도 있는 모델을 만들기엔 부족한 규모 — 데이터 증강/추가 수집 필요
- k-fold 교차 검증 등 다른 파인튜닝 기법 미시도
- 배치 사이즈 / epoch / freeze 레이어 수에 대한 체계적 탐색 없음
- YOLOv5(2020) → 최신 버전(YOLOv8/11 등) 마이그레이션 검토 여지
- 데이터 반입 후 재학습하여 본 보고서의 실험 결과 섹션을 실측치로 갱신 예정

## 11. 참고

- GitHub: https://github.com/seogy00n/AI_HUB (커밋 `1fea8e3` 저장소 재구성)
- Notion 1차 연구 보고서: https://app.notion.com/p/3967d7f5d24681129a40ce237bdad5cf
- Notion 개발 재구성 로그: https://app.notion.com/p/3977d7f5d24681c38c23ea4b75dc1278
