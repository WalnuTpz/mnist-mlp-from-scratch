# 从零实现 MNIST MLP

一个基于 PyTorch 的 MNIST 手写数字分类的多层感知机（MLP）项目：手写线性层与激活层的前向，损失函数使用数值稳定的 softmax cross entropy，反向使用 autograd，并手写优化器与训练循环。

## 目录结构
- `src/`：核心实现（数据、层、模型、损失、优化器、训练与评估）
- `scripts/`：辅助脚本（数据自检、错误样本与混淆矩阵分析）
- `data/`：MNIST 数据集目录（git 忽略）
- `checkpoints/`：模型保存目录（git 忽略）
- `logs/`：训练与分析日志目录（git 忽略）

## 环境准备
1. 创建并进入虚拟环境（可选）。
2. 安装依赖：
```bash
pip install -r requirements.txt
```

## MNIST 数据集下载

本项目默认使用 `data/` 作为 MNIST 数据根目录。首次运行时请先准备该目录并下载数据。

1. 创建目录并下载数据

```bash
mkdir -p data
python -c "from torchvision.datasets import MNIST; MNIST(root='data', train=True, download=True); MNIST(root='data', train=False, download=True)"
```

下载完成后，目录结构通常类似：

- `data/MNIST/raw/`：原始压缩文件与解压后的 IDX 文件
- `data/MNIST/processed/`：torchvision 处理后的缓存文件（如存在）

2. 验证数据可用

```bash
python -m scripts.data_sanity --data-dir data
```

## 快速开始
1. 数据自检：
```bash
python -m scripts.data_sanity --data-dir data
```
2. 训练：
```bash
python -m src.train --epochs 10
```
3. 关闭 CSV 日志：
```bash
python -m src.train --epochs 10 --no-log-csv
```

## 训练命令示例
- 基线训练（10 个 epoch）：
```bash
python -m src.train --epochs 10
```
- 自定义隐藏层大小：
```bash
python -m src.train --epochs 10 --hidden1 512 --hidden2 256
```
- 启用 BatchNorm：
```bash
python -m src.train --epochs 10 --batchnorm
```
- 调整 Dropout：
```bash
python -m src.train --epochs 10 --dropout 0.1
```
- 关闭学习率调度：
```bash
python -m src.train --epochs 10 --lr-scheduler none
```

## 训练输出
- 最佳模型：`checkpoints/best.pt`
- 训练曲线：`logs/train.csv`

## 错误样本与混淆矩阵
运行分析脚本：
```bash
python -m scripts.analyze
```
输出内容：
- 混淆矩阵：`logs/analysis/confusion_matrix.csv`
- 错误样本清单：`logs/analysis/errors.csv`
- 错误样本图像：`logs/analysis/errors/*.png`

## 常用参数
- 训练：`--epochs`、`--batch-size`、`--hidden1`、`--hidden2`、`--lr`、`--optimizer`、`--dropout`、`--batchnorm`、`--lr-scheduler`、`--warmup-epochs`、`--min-lr`
- 日志：`--log-csv`、`--no-log-csv`
- 分析：`--checkpoint`、`--out-dir`、`--max-errors`
