# PaddlePaddle-Fasttext-Embedding
基于Paddle的Fasttext词嵌入方法的实现
# Enriching Word Vectors with Subword Information
Piotr Bojanowski∗ and Edouard Grave∗ and Armand Joulin and Tomas Mikolov

fast text基本沿用了skip gram的框架。所不同的一点在于，在skip gram中训练的向量是对应的某个词，而在fast text中训练的向量则对应的是词的子结构。

什么是词的子结构？对于一个词where来说，简单地将，它由w，h，e，r四个字母组成。如果同时考虑相邻的两个字母，则它由wh，he，er，re四个部分组成。如果是考虑相邻的三个字母呢？它由whe,her，ere三部分组成。以上在原文中被称为该词的1-grams，2-grams和3-grams形式。如果单纯地这样分解的话，会出现一个问题，整词和子结构可能相同，比如where中的her和表示人称的her是相同的。但是这两种情形下的her的意义不同，一个是子结构，一个是完整的词，如果不加以区分，会给词向量的训练造成麻烦（试想字母含量少的词会是很多其他长词的子结构，肯定很不科学）。文中为了解决这个问题，很巧妙地在每个词前后加上了‘<’，‘>’两个符号，where就变成了<where>。这样之后，所有词都增加了两个符号，不影响它们正确表达自己的意思，也避免了上述问题：由于这两个符号只出现在完整的词的两端，所以不可能有一个子结构是和完整的词是相同的（试想一下原因）。

子结构有什么用呢？作者将每个词的向量表示成子结构对应的向量的和，这样知道了某个词的子结构，就可以求出该词的向量了。而在训练中，也是先将某个中心词表示为其所有子结构向量的和，然后再按照skip gram的思路进行训练，梯度的传递也传到子结构的向量上。

### fast_text_0 文件基于Paddle动态图模式编程
### fast_text_1 文件基于TensorFlow框架编程（程序思路并不好，不建议参考）
### fast_text_2 文件为纯Python编程实现
### fast_text_static 文件基于Paddle静态图模式编程
