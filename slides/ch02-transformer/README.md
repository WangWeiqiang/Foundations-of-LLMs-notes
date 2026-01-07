# 第二章：Transformer架构详解

[← 上一章：大语言模型简介](../ch01-introduction/README.md) | [返回目录](../../README.md) | [下一章：模型训练基础 →](../ch03-training/README.md)

---

## 🎯 本章学习目标

- 理解Transformer的整体架构
- 掌握自注意力机制的原理
- 理解多头注意力的作用
- 学习位置编码的必要性

---

## 🏗️ 2.1 Transformer整体架构

### 架构概览

```
           📝 输入文本
              ⬇️
        [Token Embedding]
              ⬇️
      [Position Encoding]
              ⬇️
    ┌─────────────────────┐
    │   Encoder Stack     │  ← 理解输入
    │  (N层编码器)        │
    └─────────────────────┘
              ⬇️
    ┌─────────────────────┐
    │   Decoder Stack     │  ← 生成输出
    │  (N层解码器)        │
    └─────────────────────┘
              ⬇️
        [Linear + Softmax]
              ⬇️
           🎯 输出文本
```

### 核心组件

| 组件 | 作用 | 形象理解 |
|------|------|----------|
| **Embedding** | 将词转为向量 | 给每个词一个"身份证" |
| **Position Encoding** | 记录词的位置 | 给词加上"座位号" |
| **Self-Attention** | 理解词之间关系 | 让词"互相看看" |
| **Feed-Forward** | 提取特征 | 深度思考每个词 |
| **Layer Norm** | 稳定训练 | 保持数值平衡 |

---

## 🔍 2.2 自注意力机制 (Self-Attention)

### 核心思想

**让模型关注输入中的重要信息**

```
例子：理解句子 "The animal didn't cross the street because it was too tired"

问题：'it' 指代什么？

自注意力机制帮助模型发现：
  it ←──────────── animal (关联度高！)
  it ←─ street (关联度低)
```

### 计算步骤

#### 步骤1：生成 Q, K, V 矩阵

```
📊 原理：
对于每个词，生成三个向量：
- Query (Q)：  我要查询什么？   🔍
- Key (K)：    我是什么？       🔑
- Value (V)：  我的内容是什么？ 💎

公式：
Q = X × W_Q
K = X × W_K  
V = X × W_V

其中 X 是输入，W 是权重矩阵
```

#### 步骤2：计算注意力分数

```
🧮 计算相似度：
        Q · Kᵀ
Score = ──────
        √d_k

为什么除以√d_k？
- 防止数值过大
- 保持梯度稳定
- d_k 是向量维度

形象理解：
Query 和 Key 的点积 = 两个词的相关程度
值越大 = 关系越密切
```

#### 步骤3：应用Softmax

```
📈 归一化分数：
Attention Weights = Softmax(Score)

作用：
- 将分数转为概率分布
- 总和为1
- 突出重要关系

例子：
原始分数：  [2.5, 1.0, 0.5]
Softmax后： [0.7, 0.2, 0.1]
            ⬆️   ⬆️   ⬆️
           重要 次要 不重要
```

#### 步骤4：加权求和

```
🎯 最终输出：
Output = Attention Weights × V

理解：
根据重要程度，加权组合所有词的信息
```

### 完整公式

```
         Q × Kᵀ
Attention(Q,K,V) = Softmax(──────) × V
                           √d_k
```

### 图解示例

```
输入句子："我 爱 自然语言处理"

              我    爱    自然语言处理
我      [  0.7   0.2      0.1   ]  ← 注意力权重
爱      [  0.3   0.5      0.2   ]
自然语言处理 [  0.1   0.2      0.7   ]

解读：
- "我" 最关注自己 (0.7)
- "爱" 关注自己和"我" 
- "自然语言处理" 最关注自己 (0.7)
```

---

## 🎭 2.3 多头注意力 (Multi-Head Attention)

### 为什么需要多头？

```
🤔 单头的局限：
只能关注一种关系模式

💡 多头的优势：
可以同时关注多种关系

形象比喻：
单头 = 单视角观察
多头 = 多视角观察，看得更全面
```

### 工作原理

```
                  输入
                   ⬇️
        ┌──────────┼──────────┐
        ⬇️         ⬇️         ⬇️
     Head 1    Head 2    Head 8
    (语法)    (语义)    (长距离依赖)
        ⬇️         ⬇️         ⬇️
        └──────────┬──────────┘
                [Concat]
                   ⬇️
              [Linear投影]
                   ⬇️
                  输出
```

### 数学表示

```
MultiHead(Q,K,V) = Concat(head₁,...,headₕ) × Wᴼ

其中：
headᵢ = Attention(Q×Wᵢᵠ, K×Wᵢᴷ, V×Wᵢⱽ)

参数：
- h：头的数量（通常8或16）
- 每个头关注不同的表示子空间
```

### 实例说明

```
📚 例子："The cat sat on the mat"

Head 1 关注：语法关系
  cat → sat (主谓关系)
  sat → on  (动介关系)

Head 2 关注：语义关系
  cat → mat (位置关系)
  on  → mat (空间关系)

Head 3 关注：长距离依赖
  The → cat (指代关系)

多个视角综合 → 完整理解
```

---

## 📍 2.4 位置编码 (Positional Encoding)

### 为什么需要位置信息？

```
❌ 问题：自注意力本身不考虑词的顺序

例子：
"我喜欢你" 和 "你喜欢我"
如果不考虑位置，注意力机制会给出相同结果！

✅ 解决：添加位置编码
```

### 位置编码公式

```
📐 三角函数编码：

PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

其中：
- pos：词的位置 (0, 1, 2, ...)
- i：  维度索引 (0, 1, 2, ...)
- d：  向量维度
```

### 为什么用三角函数？

```
✅ 优势：
1. 值域有界：[-1, 1]
2. 周期性：可以处理任意长度序列
3. 相对位置：PE(pos+k) 可由 PE(pos) 线性表示
4. 确定性：不需要学习参数

可视化：
位置0:  [sin(0/10000^0), cos(0/10000^0), ...]
位置1:  [sin(1/10000^0), cos(1/10000^0), ...]
位置2:  [sin(2/10000^0), cos(2/10000^0), ...]
        ⬇️
     不同频率的波形叠加
```

### 图解

```
📊 不同维度的位置编码：

维度0(高频): ∿∿∿∿∿∿∿∿∿∿
维度1(中频): ～～～～～
维度2(低频): ～～～

组合起来 = 唯一的位置标识
```

---

## 🔄 2.5 前馈神经网络 (Feed-Forward Network)

### 结构

```
🧠 两层全连接网络：

输入 (d_model)
    ⬇️
[Linear + ReLU] (扩展到 d_ff)
    ⬇️
[Linear] (压缩回 d_model)
    ⬇️
输出
```

### 公式

```
FFN(x) = max(0, x×W₁ + b₁)×W₂ + b₂

或使用GELU激活：
FFN(x) = GELU(x×W₁ + b₁)×W₂ + b₂

维度变化：
512 → 2048 → 512
(d_model → d_ff → d_model)
```

### 作用

```
💡 功能：
1. 增加模型容量
2. 非线性变换
3. 位置独立处理（每个位置用相同参数）

形象理解：
注意力  = 词与词之间交流
前馈网络 = 每个词独立思考
```

---

## 🧩 2.6 完整Transformer层

### Encoder层结构

```
       输入
        ⬇️
    [Layer Norm]
        ⬇️
 [Multi-Head Attention]
        ⬇️
   [残差连接] ← ─ ─ ┐
        ⬇️          │
    [Layer Norm]     │
        ⬇️          │
[Feed-Forward Network]
        ⬇️          │
   [残差连接] ← ─ ─ ┘
        ⬇️
       输出
```

### 残差连接 (Residual Connection)

```
🔄 作用：
Y = X + Sublayer(X)

好处：
1. 缓解梯度消失
2. 方便信息流动
3. 易于训练深层网络

形象理解：
像给信息开辟"高速公路"
即使中间处理不好，原始信息也能传递
```

### Layer Normalization

```
📊 归一化公式：

LN(x) = γ × (x - μ)/σ + β

其中：
- μ：均值
- σ：标准差
- γ, β：可学习参数

作用：
- 加速训练
- 稳定梯度
- 减少内部协变量偏移
```

---

## 🎓 2.7 关键参数配置

### 典型配置（GPT-2 Small）

```
📋 参数表：

| 参数            | 值    | 说明                |
|----------------|-------|---------------------|
| d_model        | 768   | 隐藏层维度          |
| n_heads        | 12    | 注意力头数          |
| d_ff           | 3072  | 前馈网络中间维度    |
| n_layers       | 12    | Transformer层数     |
| vocab_size     | 50257 | 词汇表大小          |
| max_seq_length | 1024  | 最大序列长度        |

总参数量：约 1.17 亿
```

### 参数量计算

```
🧮 主要参数来源：

1. Embedding层：
   vocab_size × d_model = 50257 × 768

2. 每个Attention层：
   Q,K,V 矩阵：3 × (d_model × d_model)
   输出投影：d_model × d_model

3. 每个FFN层：
   W₁：d_model × d_ff
   W₂：d_ff × d_model

4. 乘以层数：n_layers
```

---

## 💡 关键要点总结

### 核心概念

1. **Self-Attention = 让词之间互相"看看"**
   ```
   Query：我想找什么？
   Key：  我是什么？
   Value：我有什么内容？
   ```

2. **Multi-Head = 多角度观察**
   - 不同的头关注不同的关系
   - 就像多个专家一起分析

3. **Position Encoding = 给词加上"座位号"**
   - 使用三角函数
   - 让模型知道词的顺序

4. **Residual + LayerNorm = 训练稳定器**
   - 残差连接：信息高速公路
   - LayerNorm：保持数值稳定

### 架构优势

```
✅ vs RNN:
- 并行计算（更快）
- 长距离依赖（更好）

✅ vs CNN:
- 全局感受野
- 位置无关计算
```

---

## 🎯 思考题

1. **机制理解**
   - 为什么注意力分数要除以√d_k？
   - 多头注意力比单头好在哪里？

2. **设计思考**
   - 如果不用位置编码会怎样？
   - 残差连接为什么重要？

3. **实践问题**
   - 如何选择注意力头的数量？
   - 增加层数一定会提升性能吗？

---

## 📊 可视化工具推荐

- 🔗 [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) - 交互式Transformer可视化工具
- 🔗 [BertViz](https://github.com/jessevig/bertviz) - 可视化注意力权重分布
- 🔗 [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/) - 带注释的代码实现

---

## 📚 延伸阅读

- 📄 原始论文：[Attention is All You Need](https://arxiv.org/abs/1706.03762)
- 📝 博客：[The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- 📝 博客：[The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)

---

[← 上一章：大语言模型简介](../ch01-introduction/README.md) | [返回目录](../../README.md) | [下一章：模型训练基础 →](../ch03-training/README.md)
