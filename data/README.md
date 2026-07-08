# 데이터 디렉토리

이 폴더는 git에 커밋되지 않습니다 (`.gitignore` 참고). 아래 구조에 맞춰 데이터를 넣으면 `src/`, `scripts/`의 코드가 그대로 동작합니다.

## 1) 분류기(ResNet50) 학습용 — `data/classification/`

`torchvision.datasets.ImageFolder` 형식. 클래스별 폴더 아래에 이미지를 넣습니다.

```
data/classification/
├── cardboard/
├── paper/
├── metal/
├── plastic/
├── glass/
└── trash/
```

## 2) 탐지기(YOLOv5) 학습용 — `data/yolo/`

이미지와 라벨을 분리한 표준 YOLO 레이아웃. 라벨은 이미지와 같은 파일명의 `.txt` (YOLO 포맷: `class cx cy w h`, 0~1 정규화).

```
data/yolo/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

클래스 순서는 `configs/data.yaml`의 `names`와 동일해야 합니다:
`['cardboard', 'paper', 'metal', 'plastic', 'glass', 'trash']`

CVAT에서 YOLO 포맷으로 export하면 위 구조에 맞게 바로 정리할 수 있습니다.
