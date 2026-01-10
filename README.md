# 《大模型基础》（毛玉仁，高云君等著）学习笔记

<img src="assets/note-cover.png" alt="Note cover" width="480">

> # **无上甚深微妙法，百千万劫难遭遇；**
>
> # **我今见闻得受持，愿解如来真实义。**

作为一名再普通不过的软件开发工程师，我怀着求知与敬畏之心翻开此书，试图由此窥见大模型的真正奥义。然而很快便意识到，这并非一条轻松可行的道路。

书中所涉内容，往往以扎实而深厚的数学基础为前提，推导严谨、公式繁复，符号交错之间，常令人目不暇接、难辨其旨。

诚然，作者笔触流畅，已竭力将原理与技术细节化繁为简，但当一页页公式铺陈开来时，仍难免生出“知其在前，却不知其所指”的困惑。

正因如此，为了尽可能贴近这些思想的本意、把握其中的关键脉络，我尝试借助 AI 的力量，撰写这一系列解读文章，以期在反复拆解与重述之中，逐步走近大模型的核心世界。

By 王维强

阅读原书籍，请前往 https://github.com/ZJU-LLMs/Foundations-of-LLMs

## 第一章 语言模型基础

1. [n 阶马尔可夫假设 - 理想与近似](https://github.com/WangWeiqiang/Foundations-of-LLMs-notes/blob/main/%E7%AC%AC%E4%B8%80%E7%AB%A0%20%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E5%9F%BA%E7%A1%80/n%20%E9%98%B6%E9%A9%AC%E5%B0%94%E5%8F%AF%E5%A4%AB%E5%81%87%E8%AE%BE%20-%20%E7%90%86%E6%83%B3%E4%B8%8E%E8%BF%91%E4%BC%BC.pdf)
2. [极大似然估计 - 从数据到洞见](https://github.com/WangWeiqiang/Foundations-of-LLMs-notes/blob/main/%E7%AC%AC%E4%B8%80%E7%AB%A0%20%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E5%9F%BA%E7%A1%80/%E6%9E%81%E5%A4%A7%E4%BC%BC%E7%84%B6%E4%BC%B0%E8%AE%A1.pdf)
3. [前馈神经网络(FNN) - 高效但健忘的机器](https://github.com/WangWeiqiang/Foundations-of-LLMs-notes/blob/main/%E7%AC%AC%E4%B8%80%E7%AB%A0%20%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E5%9F%BA%E7%A1%80/%E5%89%8D%E9%A6%88%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%EF%BC%88FNN%EF%BC%89%20-%20%E9%AB%98%E6%95%88%E4%BD%86%E5%81%A5%E5%BF%98%E7%9A%84%E6%9C%BA%E5%99%A8.pdf)
4. [循环神经网络(RNN) - 前事不忘后事之师](<https://github.com/WangWeiqiang/Foundations-of-LLMs-notes/blob/main/%E7%AC%AC%E4%B8%80%E7%AB%A0%20%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E5%9F%BA%E7%A1%80/%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(RNN)%20-%20%E5%89%8D%E4%BA%8B%E4%B8%8D%E5%BF%98%E5%90%8E%E4%BA%8B%E4%B9%8B%E5%B8%88.pdf>)
5. [解构 RNN 语言模型 - 溺爱与纠偏如何取舍](https://github.com/WangWeiqiang/Foundations-of-LLMs-notes/blob/main/%E7%AC%AC%E4%B8%80%E7%AB%A0%20%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E5%9F%BA%E7%A1%80/%E8%A7%A3%E6%9E%84RNN%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B.pdf)
6. [注意力机制(_Attention_) - 开启上帝视角](https://github.com/WangWeiqiang/Foundations-of-LLMs-notes/blob/main/%E7%AC%AC%E4%B8%80%E7%AB%A0%20%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E5%9F%BA%E7%A1%80/%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%20-%20%E5%BC%80%E5%90%AF%E4%B8%8A%E5%B8%9D%E8%A7%86%E8%A7%92.pdf)
7. [全连接前馈层(FFL) - 模型的知识大脑](https://github.com/WangWeiqiang/Foundations-of-LLMs-notes/blob/main/%E7%AC%AC%E4%B8%80%E7%AB%A0%20%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E5%9F%BA%E7%A1%80/%E5%85%A8%E8%BF%9E%E6%8E%A5%E5%89%8D%E9%A6%88%E7%BD%91%E7%BB%9C%20-%20%E6%A8%A1%E5%9E%8B%E7%9A%84%E7%9F%A5%E8%AF%86%E5%A4%A7%E8%84%91.pdf)
8. [层正则化 - 驯服神经网络的艺术](https://github.com/WangWeiqiang/Foundations-of-LLMs-notes/blob/main/%E7%AC%AC%E4%B8%80%E7%AB%A0%20%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E5%9F%BA%E7%A1%80/%E5%B1%82%E6%AD%A3%E5%88%99%E5%8C%96%20-%20%E9%A9%AF%E6%9C%8D%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%9A%84%E8%89%BA%E6%9C%AF.pdf)
9. [残差连接 - 不忘初心的深度捷径](https://github.com/WangWeiqiang/Foundations-of-LLMs-notes/blob/main/%E7%AC%AC%E4%B8%80%E7%AB%A0%20%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E5%9F%BA%E7%A1%80/%E6%AE%8B%E5%B7%AE%E8%BF%9E%E6%8E%A5%20-%20%E4%B8%8D%E5%BF%98%E5%88%9D%E5%BF%83%E7%9A%84%E6%B7%B1%E5%BA%A6%E6%8D%B7%E5%BE%84.pdf)
10. [Transformer 架构下语言模型的训练和推演](https://github.com/WangWeiqiang/Foundations-of-LLMs-notes/blob/main/%E7%AC%AC%E4%B8%80%E7%AB%A0%20%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E5%9F%BA%E7%A1%80/Transformer%E6%9E%B6%E6%9E%84%E4%B8%8B%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E7%9A%84%E8%AE%AD%E7%BB%83%E4%B8%8E%E6%8E%A8%E6%BC%94.pdf)
11. [贪心搜索还是波束搜索 - 解码策略对决](https://github.com/WangWeiqiang/Foundations-of-LLMs-notes/blob/main/%E7%AC%AC%E4%B8%80%E7%AB%A0%20%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E5%9F%BA%E7%A1%80/%E8%B4%AA%E5%BF%83%E6%90%9C%E7%B4%A2%E8%BF%98%E6%98%AF%E6%B3%A2%E6%9D%9F%E6%90%9C%E7%B4%A2%20-%20%E8%A7%A3%E7%A0%81%E7%AD%96%E7%95%A5%E5%AF%B9%E5%86%B3.pdf)
12. [随机采样 - 语言的艺术与逻辑](https://github.com/WangWeiqiang/Foundations-of-LLMs-notes/blob/main/%E7%AC%AC%E4%B8%80%E7%AB%A0%20%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E5%9F%BA%E7%A1%80/%E9%9A%8F%E6%9C%BA%E9%87%87%E6%A0%B7%20-%20%E8%AF%AD%E8%A8%80%E7%9A%84%E8%89%BA%E6%9C%AF%E4%B8%8E%E9%80%BB%E8%BE%91.pdf)

## 第二章 大语言模型架构

1. [Transformer 三大架构比较](https://github.com/WangWeiqiang/Foundations-of-LLMs-notes/blob/main/%E7%AC%AC%E4%BA%8C%E7%AB%A0%20%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84/Transformer%20%E4%B8%89%E5%A4%A7%E6%9E%B6%E6%9E%84.pdf)
2. [注意力矩阵与掩码机制](https://github.com/WangWeiqiang/Foundations-of-LLMs-notes/blob/main/%E7%AC%AC%E4%BA%8C%E7%AB%A0%20%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84/%E6%B3%A8%E6%84%8F%E5%8A%9B%E7%9F%A9%E9%98%B5%E4%B8%8E%E6%8E%A9%E7%A0%81%E6%9C%BA%E5%88%B6.pdf)
3. [BERT 模型架构解析]()

。。。陆续更新
