import torch
from torch import nn


def softmax_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # 计算稳定版 softmax cross entropy。
    targets = targets.to(device=logits.device, dtype=torch.long)
    stable_logits = logits - logits.max(dim=1, keepdim=True).values
    logsumexp = torch.logsumexp(stable_logits, dim=1)
    log_probs = stable_logits - logsumexp.unsqueeze(1)
    nll = -log_probs.gather(1, targets.view(-1, 1)).squeeze(1)
    return nll.mean()


class SoftmaxCrossEntropy(nn.Module):
    # softmax cross entropy 损失封装。
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return softmax_cross_entropy(logits, targets)
