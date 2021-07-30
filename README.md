# DANN-MNIST-tf2
这是论文[Unsupervised Domain Adaptation by Backpropagation](https://arxiv.org/abs/1409.7495)的复现代码，并完成了MNIST与MNIST-M数据集之间的迁移训练

# 实验环境

 1. tensorflow 2.1.0
 2. opencv 3.4.5.20
 3. numpy 1.18.1
 4. cuda 10.1


# Train
首先下载[MNIST数据集](http://yann.lecun.com/exdb/mnist/)，放在项目文件的/dataset/mnist子文件夹下。之后下载[BSDS500数据集](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html#bsds500)放在项目文件的/data/BSR_bsds500.tgz路径下。

之后为了生成MNSIT-M数据集，运行create_mnistm.py脚本，命令如下：

```python
python create_mnistm.py
```
脚本运行结束后，MNIST-M数据集将保存在项目文件的/dataset/mnistm子文件夹下。
之后运行模型训练脚本train.py即可，运行命令为为：
```python
python train.py
```

# 实验结果
下面是训练过程中的相关tensorboard的相关指标在训练过程中的走势图。首先是训练误差的走势图，主要包括训练域分类误差、训练图像分类误差和训练总误差、图像分类精度和域分类精度。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200714153058738.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMwMDkxOTQ1,size_16,color_FFFFFF,t_70#pic_center)


接下来是验证过程走势图，主要包括验证域分类误差、验证图像分类误差和验证练总误差、目标域图像分类精度，目标域域分类精度。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200714153221746.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMwMDkxOTQ1,size_16,color_FFFFFF,t_70#pic_center)
