# IMDB 影评情感分类 · DistilBERT

使用 PyTorch 和 Hugging Face，微调预训练 DistilBERT，判断英文影评是正面还是负面。这是一个学习项目，重点是理解 NLP 分类的完整流程，并手动实现分类层与训练循环。

## 我的实现思路

`影评 → Tokenizer → DistilBERT → [CLS] 向量 → Dropout → 线性分类层 → 正负情感`

- **文本编码：** 将文字拆成 token 并转为编号，统一长度为 128；用 attention mask 标记有效位置和补齐位置。
- **上下文表示：** DistilBERT 结合上下文，为每个 token 生成 768 维表示。
- **分类：** 取 `[CLS]` 位置的表示，经 Dropout 和线性层得到两个类别分数。标签为 `0 = 负面`、`1 = 正面`。
- **微调：** 用交叉熵损失和 AdamW，同时更新预训练模型与分类层。
- **评估：** 每轮计算验证损失与准确率，保存验证损失最低的模型。

## 实验设置

从 IMDB 原始训练集打乱后，取 2,000 条训练、500 条验证，两部分互不重叠。打乱种子为 42；原始测试集尚未使用。

| 参数 | 设置 |
| --- | --- |
| 预训练模型 | `distilbert/distilbert-base-uncased` |
| 最大输入长度 / 批次大小 | 128 token / 8 |
| 学习率 / 权重衰减 | `2e-5` / `0.01` |
| 训练轮数 / Dropout | 2 / 0.2 |

## 实验结果

以下为原 notebook 保存的运行记录；本次整理没有重新训练。

| Epoch | Train Loss | Validation Loss | Validation Accuracy |
| --- | --- | --- | --- |
| 1 | 0.2934 | 0.3491 | 84.80% |
| 2 | 0.1276 | 0.4108 | 84.40% |

第二轮训练损失下降，但验证损失上升、准确率略降，出现了过拟合迹象。按验证损失选择时，第一轮是这次实验中更好的模型。

**我的理解：** Token 编号本身不代表语义，Transformer 会结合上下文生成表示；预训练提供起点，微调让表示适应情感分类。训练集表现变好，也需要通过验证集检查泛化能力。

## 局限与下一步

目前只做了小样本验证，84.80% 不是测试集成绩；截断也可能丢失长影评的信息。下一步可以比较 TF-IDF + Logistic Regression 基线、分析错误样本，并在选定模型后进行独立测试集评估。

## 如何运行

使用 Jupyter 打开 [project3nlp.ipynb](project3nlp.ipynb)，从上到下运行。首个代码单元会安装所需依赖，首次运行需要联网下载数据和模型。训练默认在 CPU 上执行，耗时取决于机器。

训练完成后会生成 `best_distilbert_imdb.pth`。仓库无需上传数据集或模型权重。依赖版本尚未锁定，完整训练也尚未在整理版中复跑，因此新运行的数值可能不同。

整理版补齐了原文件中未定义的 `encoded_review` 和损失演示用的 `logits`，并移除了未使用的 Kaggle 初始化代码。核心模型结构、数据划分和训练参数保持原设置。

## 参考资料

- [IMDB 数据集](https://huggingface.co/datasets/stanfordnlp/imdb)
- [DistilBERT 官方文档](https://huggingface.co/docs/transformers/model_doc/distilbert)
