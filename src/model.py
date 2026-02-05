from torch import nn

from .layers import BatchNorm1d, Dropout, Linear, ReLU


class MLP(nn.Module):
    # MLP 模型（784->256->128->10）。
    def __init__(
        self,
        in_dim: int = 784,
        hidden1: int = 256,
        hidden2: int = 128,
        out_dim: int = 10,
        dropout: float = 0.1,
        batchnorm: bool = False,
    ) -> None:
        # 初始化网络层。
        super().__init__()
        self.fc1 = Linear(in_dim, hidden1)
        # batchnorm=False 时使用直通层，避免无效计算。
        self.bn1 = BatchNorm1d(hidden1) if batchnorm else nn.Identity()
        self.act1 = ReLU()
        # dropout<=0 时使用直通层，避免无效计算。
        self.drop1 = Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.fc2 = Linear(hidden1, hidden2)
        # batchnorm=False 时使用直通层，避免无效计算。
        self.bn2 = BatchNorm1d(hidden2) if batchnorm else nn.Identity()
        self.act2 = ReLU()
        # dropout<=0 时使用直通层，避免无效计算。
        self.drop2 = Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.fc3 = Linear(hidden2, out_dim)

    def forward(self, x):
        # 前向传播，返回 logits。
        x = self.act1(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        x = self.act2(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        x = self.fc3(x)
        return x
