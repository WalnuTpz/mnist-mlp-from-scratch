from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


@dataclass(frozen=True)
class DataConfig:
    # 数据加载与 DataLoader 的配置项。
    data_dir: Path = Path("data")
    batch_size: int = 128
    num_workers: int = 0
    pin_memory: bool = False
    download: bool = False


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    # 返回 MNIST 的训练/测试 DataLoader（M0 用于形状与 dtype 检查）。
    transform = transforms.ToTensor()

    train_ds = datasets.MNIST(
        root=str(cfg.data_dir),
        train=True,
        download=cfg.download,
        transform=transform,
    )
    test_ds = datasets.MNIST(
        root=str(cfg.data_dir),
        train=False,
        download=cfg.download,
        transform=transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    return train_loader, test_loader


if __name__ == "__main__":
    from scripts.data_sanity import main

    main()
