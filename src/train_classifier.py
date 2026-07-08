"""ResNet50 전이학습으로 재활용품 분류기를 학습한다.

사용 예:
    python -m src.train_classifier --data-dir data/classification --epochs 6
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.data.dataset import (
    build_dataloaders,
    get_default_device,
    load_classification_dataset,
    to_device,
)
from src.models.resnet_classifier import ResNetClassifier


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.Adam, patience=0):
    history = []
    optimizer = opt_func(model.parameters(), lr)

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        result = evaluate(model, val_loader)
        result["train_loss"] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)

        if patience > 0:
            if result["val_loss"] < best_val_loss:
                best_val_loss = result["val_loss"]
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping at epoch {epoch + 1} (patience={patience})")
                    break

    return history


def plot_accuracies(history, out_path: Path):
    accuracies = [x["val_acc"] for x in history]
    plt.figure()
    plt.plot(accuracies, "-x")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Validation accuracy vs. epoch")
    plt.savefig(out_path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/classification")
    parser.add_argument("--output-dir", default="outputs/classifier")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=5.5e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=0,
                         help="Early stopping patience in epochs (0 = disabled)")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    device = get_default_device()
    print(f"Using device: {device}")

    splits = load_classification_dataset(
        args.data_dir, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed
    )
    print(f"classes={splits.classes} "
          f"train={len(splits.train)} val={len(splits.val)} test={len(splits.test)}")

    train_dl, val_dl, _test_dl = build_dataloaders(splits, device, batch_size=args.batch_size)

    model = to_device(ResNetClassifier(num_classes=len(splits.classes)), device)

    history = fit(
        args.epochs, args.lr, model, train_dl, val_dl,
        opt_func=torch.optim.Adam, patience=args.patience,
    )

    torch.save(model.state_dict(), output_dir / "model.bin")
    with open(output_dir / "classes.json", "w", encoding="utf-8") as f:
        json.dump(splits.classes, f, ensure_ascii=False, indent=2)
    plot_accuracies(history, output_dir / "accuracy.png")

    print(f"Saved weights, class list, and accuracy plot to {output_dir}")


if __name__ == "__main__":
    main()
