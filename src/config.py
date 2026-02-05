from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class TrainConfig:
    # 训练相关超参数与环境配置。
    data_dir: Path = Path("data")
    batch_size: int = 128
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    dropout: float = 0.1
    lr_scheduler: str = "warmup_cosine"
    warmup_epochs: int = 1
    min_lr: float = 1e-5
    seed: int = 42
    device: str = "auto"
    num_workers: int = 0
    pin_memory: bool = False
    download: bool = False


def select_device(device: str) -> torch.device:
    # 根据参数选择运行设备。
    if device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
