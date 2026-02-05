import torch


def evaluate(model, dataloader, loss_fn, device: torch.device) -> tuple[float, float]:
    # 在评估集上计算 loss 与准确率。
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    return avg_loss, acc
