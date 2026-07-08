"""ImageFolder 기반 재활용품 분류 데이터셋 유틸리티."""
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms

DEFAULT_IMAGE_SIZE = (256, 256)


def build_transforms(image_size=DEFAULT_IMAGE_SIZE):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])


@dataclass
class Splits:
    train: torch.utils.data.Dataset
    val: torch.utils.data.Dataset
    test: torch.utils.data.Dataset
    classes: list


def load_classification_dataset(data_dir, val_frac=0.1, test_frac=0.3, seed=42,
                                 image_size=DEFAULT_IMAGE_SIZE) -> Splits:
    """`data_dir` 아래 클래스별 폴더 구조(ImageFolder)를 train/val/test로 분할."""
    dataset = ImageFolder(data_dir, transform=build_transforms(image_size))

    n_total = len(dataset)
    n_val = int(n_total * val_frac)
    n_test = int(n_total * test_frac)
    n_train = n_total - n_val - n_test

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=generator)

    return Splits(train=train_ds, val=val_ds, test=test_ds, classes=dataset.classes)


def get_default_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    """DataLoader를 감싸서 배치를 꺼낼 때마다 지정한 device로 옮겨준다."""

    def __init__(self, dl: DataLoader, device: torch.device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for batch in self.dl:
            yield to_device(batch, self.device)

    def __len__(self):
        return len(self.dl)


def build_dataloaders(splits: Splits, device, batch_size=32, num_workers=4):
    train_dl = DataLoader(splits.train, batch_size, shuffle=True,
                           num_workers=num_workers, pin_memory=True)
    val_dl = DataLoader(splits.val, batch_size * 2,
                         num_workers=num_workers, pin_memory=True)
    test_dl = DataLoader(splits.test, batch_size * 2,
                          num_workers=num_workers, pin_memory=True)

    return (
        DeviceDataLoader(train_dl, device),
        DeviceDataLoader(val_dl, device),
        DeviceDataLoader(test_dl, device),
    )
