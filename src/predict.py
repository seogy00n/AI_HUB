"""단일 이미지에 대해 학습된 분류기로 예측한다.

사용 예:
    python -m src.predict --image sample.jpg \
        --checkpoint outputs/classifier/model.bin \
        --classes-json outputs/classifier/classes.json
"""
import argparse
import json

import torch
import torch.nn.functional as F
from PIL import Image

from src.data.dataset import build_transforms, get_default_device
from src.models.resnet_classifier import ResNetClassifier


def predict_image(image_path, model, classes, device):
    image = Image.open(image_path).convert("RGB")
    xb = build_transforms()(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(xb)
        probs = F.softmax(logits, dim=1)
        prob, pred = torch.max(probs, dim=1)
    return classes[pred.item()], prob.item()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", default="outputs/classifier/model.bin")
    parser.add_argument("--classes-json", default="outputs/classifier/classes.json")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_default_device()

    with open(args.classes_json, encoding="utf-8") as f:
        classes = json.load(f)

    model = ResNetClassifier(num_classes=len(classes))
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)
    model.eval()

    label, confidence = predict_image(args.image, model, classes, device)
    print(f"{args.image}: {label} ({confidence:.2%})")


if __name__ == "__main__":
    main()
