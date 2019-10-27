---
typora-root-url: ./
---

# homework1文档

[TOC]

## 代码结构

| getData.py		自己写的一些获取数据集的函数

| logistic.py		  自己写的LogisticRegression类

| part1.py			 作业2代码

| part2.py			 补充题代码

## 作业1

![IMG_20191022_225526](/asset/IMG_20191022_225526.jpg)

![IMG_20191022_225533](/asset/IMG_20191022_225533.jpg)

## 作业2

### 代码使用方法

1. python part1.py -h 可以获得使用帮助

   ![使用帮助](/asset/使用帮助.png)

2. python part1.py LDA (--rate 0.8)       括号内容可选且值为默认

3. python part1.py K-NB (--fold 5)        括号内容可选且值为默认

4. python part1.py SVM(--rate 0.8)       括号内容可选且值为默认

### 实验结果

#### LDA结果

![LDA](/asset/LDA.png)

#### K折朴素贝叶斯结果

![K折朴素贝叶斯](/asset/K折朴素贝叶斯.png)

#### SVM结果

![SVM](/asset/SVM.png)

## 补充题

### 代码使用方法

1. python part2.py -h       调用使用帮助

   ![帮助2](/asset/帮助2.png)

2. python part2.py Iris(--fold 5)       括号内容可选且值为默认

3. python part2.py watermelon(--fold 5)       括号内容可选且值为默认

   实例

   ![fold10](/asset/fold10.png)

### 实验结果

多种方法iris

![多种方法Iris](/asset/多种方法Iris.png)

多种方法西瓜![多种方法的西瓜](/asset/多种方法的西瓜.png)



## 附注

### 实验环境

sklearn.__version__ = '0.19.1'

![环境](/asset/环境.png)