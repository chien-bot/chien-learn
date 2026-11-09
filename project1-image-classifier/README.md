# Project 1 · CIFAR-10 猫狗图像分类

从 CIFAR-10 中筛选猫和狗，用 PyTorch 手写 CNN 和训练循环，并比较数据增强、权重衰减、模型结构、学习率调度和迁移学习。

> 这是猫狗二分类，`0 = cat`、`1 = dog`；不是完整的 CIFAR-10 十分类任务。

## 我的实现思路

`图片 → Tensor → CNN 特征提取 → 分类层 → 猫 / 狗`

- **数据：** 从原始训练集取 10,000 张猫狗图像，固定种子划分 8,000 张训练、2,000 张验证；只对训练集应用随机增强。
- **基础 CNN：** 两个卷积块提取特征，展平后通过全连接层输出两个类别的 logits。
- **训练：** 使用交叉熵和 Adam，手动完成清空梯度、前向计算、反向传播和参数更新。
- **模型选择：** 根据验证损失保存权重，配合 early stopping；训练结束后恢复最佳权重。
- **改进实验：** 依次尝试数据增强、Adam 的权重衰减、带 BatchNorm 的更深 CNN、StepLR，以及 ImageNet 预训练 ResNet18。

## 实验结果

以下数字来自原 notebook 的历史运行记录，本次整理未完整重训。准确率统一取**验证损失最低的那一轮**。

| 实验 | 选择的轮数 | Validation Loss | Validation Accuracy |
| --- | --- | --- | --- |
| 基础 CNN | 10 | 0.5468 | 72.65% |
| CNN + 数据增强 | 14 | 0.4873 | 75.40% |
| CNN + 数据增强 + 权重衰减 | 日志不完整 | — | — |
| ImprovedCNN + 数据增强 | 15 | 0.4139 | 80.90% |
| ImprovedCNN + 数据增强 + StepLR | 13 | 0.4145 | 80.30% |
| 预训练 ResNet18，只训练分类层 | 4 | 0.2851 | 87.70% |

权重衰减实验的原日志只到第 14 轮，缺少完整结束记录，因此不报告最终成绩。[原始训练日志](results/original_training_logs.txt) 保留了可核对的历史输出。

ResNet18 冻结了主干，只训练新的分类层，属于固定特征提取器的迁移学习；**尚未实现主干 fine-tuning**。

## 我学到了什么

卷积学习局部特征，池化缩小空间尺寸；分类层把特征转换成类别分数。训练损失下降需要结合验证结果解读，数据增强或学习率调度也不保证每次更好。预训练特征可以帮助新任务，但需要匹配预处理，并正确管理参数冻结和 BatchNorm 状态。

## 如何运行

建议 Python 3.10+，在本目录执行：

```bash
pip install -r requirements.txt
jupyter notebook CIFAR10.ipynb
```

打开 [带中文解释的 notebook](CIFAR10.ipynb)，从上到下运行。首次需联网下载 CIFAR-10 和 ResNet18 权重。有 CUDA 时自动使用 GPU，否则使用 CPU；完整运行六组实验可能较慢。

数据下载到 `data/`，训练会生成 `.pth` 权重；这些生成文件已加入忽略规则。依赖版本未锁定，不同环境下结果可能有差异。

## 检查与修正

- 修正“Dataset 本身是 tuple”的说明：单条样本才是 `(image, label)`。
- 修正平均池化的解释：缩小空间尺寸，保持通道数。
- 统一模型和批次的设备、checkpoint 的 `map_location`，并重置各实验种子和训练 loader 的生成器。
- 补充基线的最佳轮数与对应准确率，移除重复加载的 ResNet18；保留原模型结构与主要超参数。
- 清除旧输出与 Colab widget 元数据，历史数字另存为日志，避免和修改后的代码输出混淆。

整理版已在 CPU 环境（PyTorch 2.8.0、torchvision 0.23.0）通过六组实验的小样本运行检查：使用合成图片、每组 2 轮，验证标签映射、数据划分、梯度、最佳权重恢复、提前停止、学习率调度及 ResNet 主干冻结。检查时使用随机 ResNet 权重以避免下载，只验证运行逻辑，不代表预训练模型的分类效果；尚未完整重训或验证 CUDA 运行。

## 局限与下一步

目前只有一个验证划分，且多次用它选择方案，尚未使用官方独立测试集。模型结构、初始化、输入预处理和训练预算也存在差异，结果不属于严格的单变量消融实验。

下一步可以重复多个随机种子、解冻 ResNet18 的部分层进行微调、查看分类错误，并在选定方案后做独立测试集评估。

## 参考资料

- [CIFAR-10 官方数据集](https://www.cs.toronto.edu/~kriz/cifar.html)
- [PyTorch 迁移学习教程](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [torchvision ResNet18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
