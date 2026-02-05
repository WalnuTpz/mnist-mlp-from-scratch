from __future__ import annotations

import argparse
import csv
import math
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


def set_optimizer_lr(optimizer, lr: float) -> None:
    # 更新优化器学习率。
    optimizer.lr = lr


def compute_warmup_cosine_lr(
    epoch: int, base_lr: float, warmup_epochs: int, total_epochs: int, min_lr: float
) -> float:
    # 计算 warmup + cosine 的学习率。
    if warmup_epochs <= 0:
        warmup_epochs = 0
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return base_lr * epoch / warmup_epochs
    if total_epochs <= warmup_epochs:
        return base_lr
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


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


def init_csv_logger(log_path: Path) -> None:
    # 初始化 CSV 训练曲线文件（若不存在则写入表头）。
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        return
    with log_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "test_loss", "test_acc"])


def append_csv_row(
    log_path: Path, epoch: int, train_loss: float, test_loss: float, test_acc: float
) -> None:
    # 追加一行训练曲线数据到 CSV。
    with log_path.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, f"{train_loss:.6f}", f"{test_loss:.6f}", f"{test_acc:.6f}"])


def save_best_checkpoint(
    save_dir: Path,
    model,
    epoch: int,
    test_loss: float,
    test_acc: float,
    cfg: TrainConfig,
) -> Path:
    # 保存最佳模型 checkpoint。
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / "best.pt"
    payload = {
        "epoch": epoch,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "model_state": model.state_dict(),
        "train_config": cfg.__dict__,
    }
    torch.save(payload, path)
    return path


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
    parser.add_argument("--dropout", type=float, default=base_cfg.dropout)
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default=base_cfg.lr_scheduler,
        choices=["none", "warmup_cosine"],
    )
    parser.add_argument("--warmup-epochs", type=int, default=base_cfg.warmup_epochs)
    parser.add_argument("--min-lr", type=float, default=base_cfg.min_lr)
    parser.add_argument("--seed", type=int, default=base_cfg.seed)
    parser.add_argument("--device", type=str, default=base_cfg.device)
    parser.add_argument("--num-workers", type=int, default=base_cfg.num_workers)
    parser.add_argument("--pin-memory", action="store_true", default=base_cfg.pin_memory)
    parser.add_argument("--download", action="store_true", default=base_cfg.download)
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--log-csv", type=Path, default=Path("logs/train.csv"))
    parser.add_argument("--no-log-csv", action="store_true")
    args = parser.parse_args()

    cfg = TrainConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        dropout=args.dropout,
        lr_scheduler=args.lr_scheduler,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        download=args.download,
    )
    save_dir = args.save_dir
    log_csv = args.log_csv
    enable_csv = not args.no_log_csv

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

    model = MLP(dropout=cfg.dropout).to(device)
    loss_fn = SoftmaxCrossEntropy()
    optimizer = build_optimizer(cfg.optimizer, model.parameters(), cfg.lr, cfg.weight_decay)

    if enable_csv:
        init_csv_logger(log_csv)
    best_acc = 0.0
    for epoch in range(1, cfg.epochs + 1):
        if cfg.lr_scheduler == "warmup_cosine":
            lr = compute_warmup_cosine_lr(
                epoch, cfg.lr, cfg.warmup_epochs, cfg.epochs, cfg.min_lr
            )
            set_optimizer_lr(optimizer, lr)
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
        if test_acc > best_acc:
            best_acc = test_acc
            save_best_checkpoint(save_dir, model, epoch, test_loss, test_acc, cfg)
        if enable_csv:
            append_csv_row(log_csv, epoch, train_loss, test_loss, test_acc)
        print(
            "Epoch {}/{} | train loss: {:.4f} | test loss: {:.4f} | test acc: {:.2f}%".format(
                epoch, cfg.epochs, train_loss, test_loss, test_acc * 100
            )
        )


if __name__ == "__main__":
    main()
