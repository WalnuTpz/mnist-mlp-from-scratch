import math

import torch


class SGD:
    # SGD 优化器。
    def __init__(self, params, lr: float = 0.1, weight_decay: float = 0.0) -> None:
        # 初始化优化器参数。
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.weight_decay = weight_decay

    def zero_grad(self) -> None:
        # 清空参数梯度。
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self) -> None:
        # 执行一步参数更新。
        with torch.no_grad():
            for p in self.params:
                if p.grad is None:
                    continue
                grad = p.grad
                if self.weight_decay != 0.0:
                    grad = grad.add(p, alpha=self.weight_decay)
                p.add_(grad, alpha=-self.lr)


class AdamW:
    # AdamW 优化器。
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
    ) -> None:
        # 初始化优化器参数与状态。
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.state = {}

    def zero_grad(self) -> None:
        # 清空参数梯度。
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self) -> None:
        # 执行一步参数更新。
        beta1, beta2 = self.betas
        with torch.no_grad():
            for p in self.params:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW 不支持稀疏梯度")

                state = self.state.setdefault(p, {})
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                if self.weight_decay != 0.0:
                    p.add_(p, alpha=-self.lr * self.weight_decay)

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                step_size = self.lr * math.sqrt(bias_correction2) / bias_correction1

                denom = exp_avg_sq.sqrt().add_(self.eps)
                p.addcdiv_(exp_avg, denom, value=-step_size)
