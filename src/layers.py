import math

import torch
from torch import nn


class Linear(nn.Module):
    # 线性层
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        # 初始化线性层参数。
        super().__init__()
        weight = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        self.weight = nn.Parameter(weight)

        if bias:
            # 初始化 bias 参数。
            fan_in = in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            bias_param = torch.empty(out_features)
            nn.init.uniform_(bias_param, -bound, bound)
            self.bias = nn.Parameter(bias_param)
        else:
            # 不使用 bias 时注册空参数。
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 执行线性变换。
        if self.bias is None:
            return x @ self.weight.t()
        return x @ self.weight.t() + self.bias


class ReLU(nn.Module):
    # ReLU 激活层
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 应用 ReLU 激活。
        return torch.relu(x)


class Dropout(nn.Module):
    # Dropout 层（训练时随机置零）。
    def __init__(self, p: float = 0.5) -> None:
        # 初始化 Dropout 概率。
        super().__init__()
        if p < 0.0 or p >= 1.0:
            raise ValueError("Dropout 概率必须在 [0, 1) 范围内")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 训练时应用 Dropout，评估时直通。
        if self.p == 0.0 or not self.training:
            return x
        mask = torch.rand_like(x) > self.p
        return x * mask / (1.0 - self.p)
