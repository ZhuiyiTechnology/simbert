# SimBERT
基于UniLM思想、融检索与生成于一体的BERT模型。

## 模型简介

<img src="https://kexue.fm/usr/uploads/2020/05/2840550561.png" width=600 alt="SimBERT训练方式示意图.png" />

假设SENT_a和SENT_b是一组相似句，那么在同一个batch中，把<font color=blue>[CLS] SENT_a [SEP] SENT_b [SEP]</font>和<font color=blue>[CLS] SENT_b [SEP] SENT_a [SEP]</font>都加入训练，做一个相似句的生成任务，这是Seq2Seq部分。

另一方面，把整个batch内的<font color=blue>[CLS]</font>向量都拿出来，得到一个句向量矩阵$\boldsymbol{V}\in\mathbb{R}^{b\times d}$（$b$是batch_size，$d$是hidden_size），然后对$d$维度做$l_2$归一化，得到$\tilde{\boldsymbol{V}}$，然后两两做内积，得到$b\times b$的相似度矩阵$\tilde{\boldsymbol{V}}\tilde{\boldsymbol{V}}^{\top}$，接着乘以一个scale（我们取了30），并mask掉对角线部分，最后每一行进行softmax，作为一个分类任务训练，每个样本的目标标签是它的相似句（至于自身已经被mask掉）。说白了，就是把batch内所有的非相似样本都当作负样本，借助softmax来增加相似样本的相似度，降低其余样本的相似度。

详细介绍请看：[https://kexue.fm/archives/7427](https://kexue.fm/archives/7427)

## 训练环境
tensorflow 1.14 + keras 2.3.1 + bert4keras 0.7.7

## 如何引用

Bibtex：

```tex
@techreport{simbert,
  title={SimBERT: Integrating Retrieval and Generation into BERT},
  author={Jianlin Su},
  year={2020},
  url = "https://github.com/ZhuiyiTechnology/simbert",
}
```
