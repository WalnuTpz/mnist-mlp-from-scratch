from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torchvision.utils import save_image

from src.config import select_device
from src.data import DataConfig, MNIST_MEAN, MNIST_STD, get_dataloaders
from src.model import MLP


def _denormalize_flat(x_flat: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    # 反归一化并恢复为 [1, 28, 28]。
    img = x_flat.view(1, 28, 28)
    img = img * std + mean
    return img.clamp(0.0, 1.0)


def _update_confusion(confusion: torch.Tensor, preds: torch.Tensor, targets: torch.Tensor) -> None:
    # 更新混淆矩阵计数。
    for t, p in zip(targets.view(-1), preds.view(-1)):
        confusion[int(t), int(p)] += 1


def _save_error_image(
    errors_dir: Path,
    index: int,
    image: torch.Tensor,
    true_label: int,
    pred_label: int,
) -> Path:
    # 保存错误样本图像。
    filename = f"{index:05d}_t{true_label}_p{pred_label}.png"
    path = errors_dir / filename
    save_image(image, str(path))
    return path


def _write_confusion_csv(path: Path, confusion: torch.Tensor) -> None:
    # 将混淆矩阵写入 CSV。
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["true\\pred"] + [f"pred_{i}" for i in range(confusion.size(1))]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(confusion.size(0)):
            row = [f"true_{i}"] + [int(v) for v in confusion[i].tolist()]
            writer.writerow(row)


def _write_errors_csv(path: Path, rows: list[list[str]]) -> None:
    # 将错误样本记录写入 CSV。
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "true", "pred", "image_path"])
        writer.writerows(rows)


def main() -> None:
    # 载入模型并生成混淆矩阵与错误样本。
    parser = argparse.ArgumentParser(description="MNIST 错误样本与混淆矩阵分析")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("logs/analysis"))
    parser.add_argument("--max-errors", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    device = select_device(args.device)
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"未找到 checkpoint: {args.checkpoint}")

    data_cfg = DataConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        download=args.download,
    )
    _, test_loader = get_dataloaders(data_cfg)

    model = MLP().to(device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    model.eval()

    confusion = torch.zeros(10, 10, dtype=torch.long)
    errors_dir = args.out_dir / "errors"
    errors_dir.mkdir(parents=True, exist_ok=True)
    error_rows: list[list[str]] = []

    mean = torch.tensor(MNIST_MEAN).view(1, 1, 1)
    std = torch.tensor(MNIST_STD).view(1, 1, 1)

    total_samples = 0
    total_errors = 0
    saved_errors = 0
    sample_offset = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)

            _update_confusion(confusion, preds.detach().cpu(), y.detach().cpu())

            mis_mask = preds.ne(y)
            mis_indices = mis_mask.nonzero(as_tuple=False).view(-1)
            total_errors += int(mis_indices.numel())

            if args.max_errors > 0 and saved_errors < args.max_errors:
                for i in mis_indices.tolist():
                    if saved_errors >= args.max_errors:
                        break
                    flat = x[i].detach().cpu()
                    img = _denormalize_flat(flat, mean, std)
                    true_label = int(y[i].item())
                    pred_label = int(preds[i].item())
                    path = _save_error_image(
                        errors_dir, sample_offset + i, img, true_label, pred_label
                    )
                    error_rows.append(
                        [str(sample_offset + i), str(true_label), str(pred_label), str(path)]
                    )
                    saved_errors += 1

            batch_size = x.size(0)
            total_samples += batch_size
            sample_offset += batch_size

    acc = 1.0 - total_errors / total_samples if total_samples > 0 else 0.0
    _write_confusion_csv(args.out_dir / "confusion_matrix.csv", confusion)
    _write_errors_csv(args.out_dir / "errors.csv", error_rows)

    print(f"分析完成：样本数 {total_samples}，错误数 {total_errors}，准确率 {acc:.4f}")
    print(f"混淆矩阵已保存：{args.out_dir / 'confusion_matrix.csv'}")
    print(f"错误样本已保存：{args.out_dir / 'errors.csv'}，图像在 {errors_dir}")


if __name__ == "__main__":
    main()
