from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.data import DataConfig, get_dataloaders


def _print_batch_stats(x: torch.Tensor, y: torch.Tensor) -> None:
    # 打印 batch 的形状与数据类型。
    print(f"x shape: {tuple(x.shape)}")
    print(f"y shape: {tuple(y.shape)}")
    print(f"x dtype: {x.dtype}")
    print(f"y dtype: {y.dtype}")
    print(f"x min: {x.min().item():.4f}")
    print(f"x max: {x.max().item():.4f}")
    print(f"x mean: {x.mean().item():.4f}")
    print(f"x std: {x.std().item():.4f}")


def main() -> None:
    # 解析参数并执行 M0 数据批次检查。
    parser = argparse.ArgumentParser(description="MNIST 数据自检（M0）")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    cfg = DataConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        download=args.download,
    )

    train_loader, test_loader = get_dataloaders(cfg)

    x, y = next(iter(train_loader))
    print("Train batch:")
    _print_batch_stats(x, y)

    x_t, y_t = next(iter(test_loader))
    print("Test batch:")
    _print_batch_stats(x_t, y_t)


if __name__ == "__main__":
    main()
