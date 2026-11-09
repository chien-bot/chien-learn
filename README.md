# chien-learn

记录我的机器学习学习项目，用代码、简短解释和实验结果展示我的理解。

## 项目

| 项目 | 内容 | 主要工具 |
| --- | --- | --- |
| [Project 3 · NLP 影评情感分类](project3-nlp/) | 微调 DistilBERT，判断 IMDB 英文影评的正负情感 | PyTorch、Hugging Face |

### Project 3：我学到了什么

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
