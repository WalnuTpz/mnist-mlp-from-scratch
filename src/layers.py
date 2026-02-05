import math

import torch
from torch import nn


class Linear(nn.Module):
    # 线性层
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        weight = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        self.weight = nn.Parameter(weight)

        if bias:
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
        return torch.relu(x)
