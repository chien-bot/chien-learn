# chien-learn

记录我的深度学习学习路线，用代码、简短解释和实验结果展示我的理解。

## 学习路线

| Project | 项目 | 核心学习内容 | 状态 |
| --- | --- | --- | --- |
| 1 | [Image Classifier](project1-image-classifier/) | CNN、Training Loop、Data Augmentation、Transfer Learning；Fine-tuning 待实现 | 基本完成 |
| 2 | Neural Network from Scratch | MLP、Backpropagation、Gradient Descent、Autograd | **下一项目** |
| 3 | [NLP Transformer Model](project3-nlp/) | Tokenization、Embedding、Transformer、Fine-tuning | 已有初版 |
| 4 | Recommendation System | Collaborative Filtering、Embeddings | 未开始 |
| 5 | Mini Deep Learning Framework | Dataset、Optimizer、Training Loop、Callbacks | 未开始 |
| 6 | CNN / ResNet / U-Net | 深层 CNN、Residual Connections、Image Segmentation | 未开始 |
| 7 | Mini Diffusion Model | DDPM、Noise Scheduling、Denoising | 未开始 |
| 8 | Stable Diffusion Project | U-Net、Attention、Latent Diffusion、生成式 AI | 未开始 |

## Project 1 · Image Classifier

从 CIFAR-10 筛选猫和狗，手写 CNN 和训练循环，再比较数据增强、权重衰减、改进 CNN、学习率调度和预训练 ResNet18。

- 基础 CNN 在最低验证损失对应轮次的准确率为 72.65%。
- 冻结 ResNet18 主干、只训练分类层，历史最佳验证损失对应的准确率为 87.70%。
- 已实现固定特征提取器的迁移学习；主干 Fine-tuning 和独立测试集评估尚未完成。

查看 [项目说明与运行方法](project1-image-classifier/README.md) 或 [带中文解释的 notebook](project1-image-classifier/CIFAR10.ipynb)。以上数字来自原始实验记录，整理版尚未完整重训。

## Project 3 · NLP Transformer Model

使用 PyTorch 和 Hugging Face 微调 DistilBERT，判断 IMDB 英文影评的正负情感。

### 我学到了什么

- 如何将文本转换成 token 编号，并用 attention mask 处理补齐位置。
- 如何使用 Transformer 的上下文表示完成情感分类。
- 如何添加分类层，并微调预训练模型。
- 如何用验证损失与准确率观察效果，识别过拟合迹象。

查看 [项目说明](project3-nlp/README.md) 或 [带中文解释的 notebook](project3-nlp/project3nlp.ipynb)。

## 运行 Project 3

在 Python 环境中进入 `project3-nlp` 目录：

```bash
pip install -r requirements.txt
jupyter notebook project3nlp.ipynb
```

打开 notebook 后从上到下运行。首次运行需要联网下载数据和模型，训练默认使用 CPU。

实验结果来自原 notebook 的运行记录，整理版本尚未重新训练；详细设置、结果和局限见项目说明。
