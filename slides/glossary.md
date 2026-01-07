# 术语表 (Glossary)

[← 返回目录](../../README.md)

本术语表收录了大语言模型相关的核心概念和专业术语，帮助快速查找和理解。

---

## A

### Adapter（适配器）
一种参数高效微调方法，在Transformer层中插入小型瓶颈模块，只训练这些新增模块。

### Adam
一种自适应学习率优化算法，结合了Momentum和RMSprop的优点。

### AdamW
Adam的改进版本，正确实现了权重衰减，是大模型训练的首选优化器。

### Attention（注意力）
一种机制，让模型能够关注输入中的重要信息，是Transformer架构的核心。

### Autoregressive（自回归）
逐个生成序列元素的方式，每次预测基于之前的所有元素。

---

## B

### Backpropagation（反向传播）
神经网络训练的核心算法，通过链式法则计算损失函数对参数的梯度。

### Batch Size（批次大小）
一次训练迭代中使用的样本数量。

### BERT (Bidirectional Encoder Representations from Transformers)
一种双向编码器模型，使用掩码语言模型进行预训练。

### BPE (Byte Pair Encoding)
一种Token化方法，通过迭代合并最频繁的字节对来构建词汇表。

---

## C

### Chain-of-Thought (CoT)（思维链）
一种Prompt技术，引导模型展示逐步推理过程。

### Checkpoint（检查点）
训练过程中保存的模型状态，包括参数、优化器状态等。

### Context Window（上下文窗口）
模型一次能处理的最大Token数量。

### Cross-Entropy Loss（交叉熵损失）
衡量预测概率分布与真实分布差异的损失函数，语言模型常用。

---

## D

### Decoder（解码器）
Transformer的一部分，用于生成输出序列。

### Dropout
一种正则化技术，训练时随机关闭部分神经元。

---

## E

### Embedding（嵌入）
将离散符号（如词）映射到连续向量空间。

### Encoder（编码器）
Transformer的一部分，用于理解输入序列。

### Epoch
完整遍历一次训练数据集。

---

## F

### Feed-Forward Network (FFN)（前馈网络）
Transformer层中的全连接网络，进行位置独立的非线性变换。

### Few-Shot Learning（少样本学习）
给模型提供少量示例即可完成新任务。

### Fine-Tuning（微调）
在预训练模型基础上，用特定任务数据继续训练。

---

## G

### Gradient（梯度）
损失函数对参数的导数，指示参数更新方向。

### Gradient Clipping（梯度裁剪）
限制梯度的最大范数，防止梯度爆炸。

### GPT (Generative Pre-trained Transformer)
一系列自回归语言模型，由OpenAI开发。

---

## H

### Hallucination（幻觉）
模型生成不真实或不存在的信息。

### Head（注意力头）
多头注意力中的一个独立注意力机制。

### Hidden Layer（隐藏层）
神经网络中输入层和输出层之间的层。

---

## I

### Inference（推理）
使用训练好的模型进行预测的过程。

---

## K

### Key (K)
自注意力机制中的键向量，用于与Query计算相似度。

---

## L

### Layer Normalization
归一化技术，在每一层对激活值进行标准化。

### Learning Rate（学习率）
控制参数更新步长的超参数。

### LLM (Large Language Model)（大语言模型）
参数量达到数十亿或更多的语言模型。

### LoRA (Low-Rank Adaptation)（低秩适配）
一种PEFT方法，通过低秩矩阵分解实现高效微调。

### Loss Function（损失函数）
衡量模型预测与真实标签差异的函数。

---

## M

### Masked Language Model (MLM)（掩码语言模型）
BERT使用的预训练任务，预测被掩码的词。

### Momentum（动量）
优化器中的技术，累积历史梯度方向。

### Multi-Head Attention（多头注意力）
并行使用多个注意力机制，捕捉不同类型的关系。

---

## N

### NLP (Natural Language Processing)（自然语言处理）
让计算机理解和生成人类语言的技术。

### Normalization（归一化）
将数据或激活值缩放到特定范围的技术。

---

## O

### Optimizer（优化器）
更新模型参数的算法，如SGD、Adam等。

### Overfitting（过拟合）
模型在训练集上表现好，但在测试集上表现差。

---

## P

### Parameter（参数）
模型中需要学习的权重和偏置。

### PEFT (Parameter-Efficient Fine-Tuning)（参数高效微调）
只更新少量参数的微调方法。

### Perplexity (PPL)（困惑度）
评估语言模型的指标，越低越好。

### Position Encoding（位置编码）
为序列中的每个位置添加位置信息。

### Pre-training（预训练）
在大规模无标注数据上训练模型的阶段。

### Prefix Tuning（前缀调优）
在输入前添加可训练的虚拟Token。

### Prompt
给模型的输入指令或问题。

### Prompt Engineering（提示工程）
设计有效Prompt的技术和方法。

---

## Q

### Query (Q)
自注意力机制中的查询向量。

---

## R

### Rank（秩）
矩阵的秩，LoRA中的关键超参数。

### Residual Connection（残差连接）
将层的输入直接加到输出上，缓解梯度消失。

### RLHF (Reinforcement Learning from Human Feedback)
从人类反馈中学习，用于对齐模型行为。

---

## S

### Self-Attention（自注意力）
序列内部元素之间相互关注的机制。

### Softmax
将实数向量转换为概率分布的函数。

### SGD (Stochastic Gradient Descent)（随机梯度下降）
基础的优化算法。

---

## T

### Token
文本的基本处理单元，可以是词、子词或字符。

### Tokenization（Token化）
将文本分割成Token的过程。

### Transfer Learning（迁移学习）
将一个任务学到的知识应用到另一个任务。

### Transformer
一种基于自注意力机制的神经网络架构。

---

## V

### Value (V)
自注意力机制中的值向量，包含实际内容。

### Vocabulary（词汇表）
模型可以处理的所有Token的集合。

---

## W

### Warmup
训练初期逐渐增加学习率的策略。

### Weight Decay（权重衰减）
一种正则化技术，防止权重过大。

---

## Z

### Zero-Shot Learning（零样本学习）
不提供示例，直接让模型完成任务。

---

## 中英对照

| 中文 | 英文 | 简写 |
|------|------|------|
| 大语言模型 | Large Language Model | LLM |
| 自然语言处理 | Natural Language Processing | NLP |
| 注意力机制 | Attention Mechanism | - |
| 自注意力 | Self-Attention | - |
| 多头注意力 | Multi-Head Attention | MHA |
| 前馈神经网络 | Feed-Forward Network | FFN |
| 位置编码 | Positional Encoding | PE |
| 残差连接 | Residual Connection | - |
| 层归一化 | Layer Normalization | - |
| 预训练 | Pre-training | - |
| 微调 | Fine-tuning | FT |
| 参数高效微调 | Parameter-Efficient Fine-Tuning | PEFT |
| 低秩适配 | Low-Rank Adaptation | LoRA |
| 提示工程 | Prompt Engineering | - |
| 思维链 | Chain-of-Thought | CoT |
| 零样本学习 | Zero-Shot Learning | - |
| 少样本学习 | Few-Shot Learning | - |
| 交叉熵 | Cross-Entropy | - |
| 困惑度 | Perplexity | PPL |
| 梯度下降 | Gradient Descent | GD |
| 反向传播 | Backpropagation | BP |
| 学习率 | Learning Rate | LR |
| 批次大小 | Batch Size | - |
| 过拟合 | Overfitting | - |
| 欠拟合 | Underfitting | - |
| 正则化 | Regularization | - |
| 权重衰减 | Weight Decay | - |
| 梯度裁剪 | Gradient Clipping | - |

---

## 公式符号说明

| 符号 | 含义 |
|------|------|
| Q | Query（查询矩阵） |
| K | Key（键矩阵） |
| V | Value（值矩阵） |
| d | 维度 |
| h | 注意力头数 |
| L | 层数 |
| η | 学习率 |
| θ | 参数 |
| ∇L | 损失的梯度 |
| β | 动量系数 |
| λ | 权重衰减系数 |
| ε | 数值稳定项 |
| σ | 标准差 |
| μ | 均值 |

---

[← 返回目录](../../README.md)
