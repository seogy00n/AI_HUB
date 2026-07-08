"""학습된 분류기를 test split에 대해 평가하고 confusion matrix를 저장한다.

사용 예:
    python -m src.evaluate_classifier --data-dir data/classification \
        --checkpoint outputs/classifier/model.bin
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from src.data.dataset import build_dataloaders, get_default_device, load_classification_dataset, to_device
from src.models.resnet_classifier import ResNetClassifier


@torch.no_grad()
def collect_predictions(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in test_loader:
        outputs = model(images)
        _, preds = torch.max(outputs, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    return all_preds, all_labels


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/classification")
    parser.add_argument("--checkpoint", default="outputs/classifier/model.bin")
    parser.add_argument("--classes-json", default="outputs/classifier/classes.json")
    parser.add_argument("--output-dir", default="outputs/classifier")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_default_device()

    with open(args.classes_json, encoding="utf-8") as f:
        classes = json.load(f)

    splits = load_classification_dataset(
        args.data_dir, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed
    )
    _train_dl, _val_dl, test_dl = build_dataloaders(splits, device, batch_size=args.batch_size)

    model = ResNetClassifier(num_classes=len(classes))
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = to_device(model, device)

    preds, labels = collect_predictions(model, test_dl)
    acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)
    print(f"Test accuracy: {acc:.4f} ({len(labels)} samples)")

    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, xticks_rotation=45)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "confusion_matrix.png")
    print(f"Saved confusion matrix to {output_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
