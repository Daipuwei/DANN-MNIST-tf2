# -*- coding: utf-8 -*-
# @Time    : 2020/7/10 12:36
# @Author  : DaiPuwei
# @Email   : 771830171@qq.com
# @File    : dann.py
# @Software: PyCharm

import tensorflow as tf

def build_feature_extractor():
    """
    这是特征提取子网络的构建函数
    :param image_input: 图像输入张量
    :param name: 输出特征名称
    :return:
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=32, kernel_size=5,strides=1),
        #tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
        tf.keras.layers.Conv2D(filters=48, kernel_size=5,strides=1),
        #tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
        tf.keras.layers.Flatten(),
    ])
    return model

def build_image_classify_extractor():
    """
    这是搭建图像分类器模型的函数
    :param image_classify_feature: 图像分类特征张量
    :return:
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(100),
        #tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        #tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(100,activation='relu'),
        #tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10,activation='softmax',name="image_cls_pred"),
    ])
    return model

def build_domain_classify_extractor():
    """
    这是搭建域分类器的函数
    :param domain_classify_feature: 域分类特征张量
    :return:
    """
    # 搭建域分类器
    model = tf.keras.Sequential([
        #GradientReversalLayer(),
        tf.keras.layers.Dense(100),
        #tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        #tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2, activation='softmax', name="domain_cls_pred")
    ])
    return model
