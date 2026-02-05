from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import TrainConfig, select_device
from .data import DataConfig, get_dataloaders
from .eval import evaluate
from .loss import SoftmaxCrossEntropy
from .model import MLP
from .optim import AdamW, SGD
from .utils import set_seed


def build_optimizer(name: str, params, lr: float, weight_decay: float):
    # 根据名称创建优化器实例。
    name = name.lower()
    if name == "sgd":
        return SGD(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"未知优化器: {name}")


def train_one_epoch(model, dataloader, loss_fn, optimizer, device: torch.device) -> float:
    # 训练一个 epoch 并返回平均 loss。
    model.train()
    total_loss = 0.0
    total_samples = 0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def main() -> None:
    # 解析参数并执行训练。
    base_cfg = TrainConfig()
    parser = argparse.ArgumentParser(description="MNIST MLP 训练（M5）")
    parser.add_argument("--data-dir", type=Path, default=base_cfg.data_dir)
    parser.add_argument("--batch-size", type=int, default=base_cfg.batch_size)
    parser.add_argument("--epochs", type=int, default=base_cfg.epochs)
    parser.add_argument("--lr", type=float, default=base_cfg.lr)
    parser.add_argument("--weight-decay", type=float, default=base_cfg.weight_decay)
    parser.add_argument("--optimizer", type=str, default=base_cfg.optimizer, choices=["sgd", "adamw"])
    parser.add_argument("--seed", type=int, default=base_cfg.seed)
    parser.add_argument("--device", type=str, default=base_cfg.device)
    parser.add_argument("--num-workers", type=int, default=base_cfg.num_workers)
    parser.add_argument("--pin-memory", action="store_true", default=base_cfg.pin_memory)
    parser.add_argument("--download", action="store_true", default=base_cfg.download)
    args = parser.parse_args()

    cfg = TrainConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        download=args.download,
    )

    set_seed(cfg.seed)
    device = select_device(cfg.device)

    data_cfg = DataConfig(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        download=cfg.download,
    )
    train_loader, test_loader = get_dataloaders(data_cfg)

    model = MLP().to(device)
    loss_fn = SoftmaxCrossEntropy()
    optimizer = build_optimizer(cfg.optimizer, model.parameters(), cfg.lr, cfg.weight_decay)

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
        print(
            "Epoch {}/{} | train loss: {:.4f} | test loss: {:.4f} | test acc: {:.2f}%".format(
                epoch, cfg.epochs, train_loss, test_loss, test_acc * 100
            )
        )


if __name__ == "__main__":
    main()
