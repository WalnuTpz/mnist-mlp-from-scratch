# MNIST MLP 从零实现（PyTorch）

一个基于 PyTorch 的“从零实现”风格 MNIST MLP 项目。手写线性层与激活层的前向，损失函数使用数值稳定的 softmax cross entropy，反向使用 autograd，并手写优化器与训练循环。

## 目录结构
- `src/`：核心实现（数据、层、模型、损失、优化器、训练与评估）
- `scripts/`：辅助脚本（数据自检、错误样本与混淆矩阵分析）
- `data/`：MNIST 数据集目录（本地已下载）
- `checkpoints/`：模型保存目录（git 忽略）
- `logs/`：训练与分析日志目录（git 忽略）

## 环境准备
1. 创建并进入虚拟环境（可选）。
2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 快速开始
1. M0/M1 数据自检：
```bash
python -m scripts.data_sanity --data-dir data
```
2. 训练（默认 AdamW + warmup+cosine）：
```bash
python -m src.train --epochs 10
```
3. 关闭 CSV 日志：
```bash
python -m src.train --epochs 10 --no-log-csv
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
- 训练：`--epochs`、`--batch-size`、`--lr`、`--optimizer`、`--lr-scheduler`、`--warmup-epochs`、`--min-lr`
- 日志：`--log-csv`、`--no-log-csv`
- 分析：`--checkpoint`、`--out-dir`、`--max-errors`
