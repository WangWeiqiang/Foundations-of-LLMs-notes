# 快速参考指南

[← 返回目录](../../README.md)

---

## 📖 核心公式速查

### 自注意力机制

```
Attention(Q, K, V) = Softmax(QK^T / √d_k) × V
```

**参数说明：**
- Q (Query): 查询矩阵
- K (Key): 键矩阵  
- V (Value): 值矩阵
- d_k: 键向量的维度

---

### 多头注意力

```
MultiHead(Q,K,V) = Concat(head₁,...,headₕ) × W^O

where headᵢ = Attention(QW^Q_i, KW^K_i, VW^V_i)
```

---

### 位置编码

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

---

### 交叉熵损失

```
Loss = -Σ y_i × log(ŷ_i)
```

---

### 困惑度

```
Perplexity = exp(Loss)
```

---

### Adam优化器

```
m_t = β₁ × m_{t-1} + (1-β₁) × ∇L
v_t = β₂ × v_{t-1} + (1-β₂) × (∇L)²

θ_{t+1} = θ_t - η × m̂_t / (√v̂_t + ε)
```

**推荐参数：**
- β₁ = 0.9
- β₂ = 0.999  
- ε = 1e-8
- η = 1e-4 到 5e-4

---

### LoRA

```
W' = W + BA

where:
- W: d×d (冻结)
- B: d×r
- A: r×d  
- r << d
```

---

## 🎯 训练配置速查

### GPT-2 配置

| 模型 | 层数 | 隐藏维度 | 注意力头 | 参数量 |
|------|------|----------|----------|--------|
| Small | 12 | 768 | 12 | 117M |
| Medium | 24 | 1024 | 16 | 345M |
| Large | 36 | 1280 | 20 | 774M |
| XL | 48 | 1600 | 25 | 1.5B |

---

### 推荐超参数

#### 预训练

```yaml
优化器: AdamW
学习率: 6e-4
Warmup: 2000 steps
调度: Cosine decay
Batch size: 0.5M tokens
梯度裁剪: 1.0
权重衰减: 0.1
Dropout: 0.1
```

#### 微调

```yaml
优化器: AdamW
学习率: 2e-5 到 5e-5
Warmup: 前10%步数
Epochs: 3-5
Batch size: 8-32
梯度裁剪: 1.0
权重衰减: 0.01
```

#### LoRA微调

```yaml
学习率: 3e-4
LoRA rank (r): 8
LoRA alpha: 16
LoRA dropout: 0.05
Target modules: q_proj, v_proj
```

---

## 📊 性能指标

### 困惑度 (Perplexity)

| 范围 | 质量 |
|------|------|
| < 10 | 优秀 |
| 10-20 | 良好 |
| 20-50 | 一般 |
| > 50 | 较差 |

---

### BLEU分数 (机器翻译)

| 分数 | 质量 |
|------|------|
| > 40 | 优秀 |
| 30-40 | 良好 |
| 20-30 | 可用 |
| < 20 | 较差 |

---

## 🔧 常见问题排查

### 训练不稳定

**症状：** Loss震荡、NaN

**解决方案：**
1. 降低学习率
2. 增加梯度裁剪 (1.0 → 0.5)
3. 减小batch size
4. 检查数据质量

---

### 过拟合

**症状：** 训练loss↓，验证loss↑

**解决方案：**
1. Early stopping
2. 增加Dropout
3. 增加数据
4. 减少训练轮数
5. 增加权重衰减

---

### 欠拟合

**症状：** 训练和验证loss都很高

**解决方案：**
1. 增加模型容量
2. 训练更长时间
3. 提高学习率
4. 减少正则化

---

### GPU内存不足

**解决方案：**
1. 减小batch size
2. 使用梯度累积
3. 使用混合精度训练
4. 使用梯度检查点
5. 考虑LoRA等PEFT方法

---

## 🎨 Prompt模板

### 文本分类

```
请对以下文本进行分类。

类别：[类别1, 类别2, 类别3]

文本：{text}

分类结果：
```

---

### 问答

```
请根据以下上下文回答问题。如果无法从上下文中找到答案，请回答"无法确定"。

上下文：{context}

问题：{question}

答案：
```

---

### 摘要

```
请用一段话总结以下文章的核心内容（不超过100字）。

文章：
{article}

摘要：
```

---

### 翻译

```
请将以下文本从{source_lang}翻译成{target_lang}，保持原文的语气和风格。

原文：{text}

译文：
```

---

### Few-shot学习

```
以下是一些示例：

示例1：
输入：{example1_input}
输出：{example1_output}

示例2：
输入：{example2_input}
输出：{example2_output}

现在请处理：
输入：{test_input}
输出：
```

---

## 🔗 有用的工具

### 训练框架
- 🔗 [Transformers (HuggingFace)](https://github.com/huggingface/transformers)
- 🔗 [DeepSpeed](https://github.com/microsoft/DeepSpeed)
- 🔗 [PyTorch](https://pytorch.org/)

### PEFT库
- 🔗 [PEFT (HuggingFace)](https://github.com/huggingface/peft)
- 🔗 [LoRA Implementation](https://github.com/microsoft/LoRA)

### 可视化工具
- 🔗 [TensorBoard](https://www.tensorflow.org/tensorboard)
- 🔗 [Weights & Biases](https://wandb.ai/)

### 数据集
- 🔗 [HuggingFace Datasets](https://huggingface.co/datasets)
- 🔗 [Common Crawl](https://commoncrawl.org/)

---

## 📱 速记卡片

### Transformer的6个核心组件

1. **Token Embedding** - 词向量
2. **Position Encoding** - 位置编码
3. **Self-Attention** - 自注意力
4. **Multi-Head Attention** - 多头注意力
5. **Feed-Forward Network** - 前馈网络
6. **Layer Normalization** - 层归一化

---

### 训练的4个阶段

1. **预训练** (Pre-training) - 通用能力
2. **微调** (Fine-tuning) - 任务适配
3. **对齐** (Alignment) - 人类偏好
4. **部署** (Deployment) - 实际应用

---

### PEFT的4种方法

1. **LoRA** - 低秩适配 (推荐)
2. **Adapter** - 适配器层
3. **Prefix Tuning** - 前缀调优
4. **Prompt Tuning** - 提示调优

---

[← 返回目录](../../README.md)
