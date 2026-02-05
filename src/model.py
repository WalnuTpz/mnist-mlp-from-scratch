from torch import nn

from .layers import Linear, ReLU


class MLP(nn.Module):
    # MLP 模型（784->256->128->10）。
    def __init__(self, in_dim: int = 784, hidden1: int = 256, hidden2: int = 128, out_dim: int = 10) -> None:
        # 初始化网络层。
        super().__init__()
        self.fc1 = Linear(in_dim, hidden1)
        self.act1 = ReLU()
        self.fc2 = Linear(hidden1, hidden2)
        self.act2 = ReLU()
        self.fc3 = Linear(hidden2, out_dim)

    def forward(self, x):
        # 前向传播，返回 logits。
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.fc3(x)
        return x
