"""ImageNet 사전학습 ResNet50 기반 재활용품 분류기 (전이학습)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        return F.cross_entropy(out, labels)

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {"val_loss": loss.detach(), "val_acc": acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x["val_acc"] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {"val_loss": epoch_loss.item(), "val_acc": epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch {}: train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch + 1, result["train_loss"], result["val_loss"], result["val_acc"]))


class ResNetClassifier(ImageClassificationBase):
    """마지막 FC 레이어를 클래스 수에 맞게 교체한 ResNet50."""

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.network = models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, xb):
        # cross_entropy는 raw logits을 기대하므로 activation을 씌우지 않는다.
        # (기존 Colab 코드는 여기에 sigmoid를 적용했는데, multi-class 분류에는
        # 맞지 않는 설정이라 이번 재구성에서 제거함 — 자세한 내용은 README 참고)
        return self.network(xb)
