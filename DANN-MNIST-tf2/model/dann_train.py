# -*- coding: utf-8 -*-
# @Time    : 2020/7/10 12:40
# @Author  : DaiPuwei
# @Email   : 771830171@qq.com
# @File    : dann_train.py
# @Software: PyCharm

import os
import datetime
import numpy as np

import tensorflow as tf
from tensorflow import keras

from model.dann import build_featur_extractor
from model.dann import build_image_classify_model
from model.dann import build_domain_classify_model

from utils.utils import grl_lambda_schedule
from utils.utils import learning_rate_schedule

class MNIST2MNIST_M_DANN(object):

    def __init__(self, config):
        """
        这是MNINST与MNIST_M域适配网络的初始化函数
        :param config: 参数配置类
        """
        # 初始化参数类
        self.cfg = config

        # 定义相关占位符
        self.grl_lambd = 1.0  # GRL层参数

        # 搭建深度域适配网络
        self.build_DANN()

        # 定义训练和验证损失与指标
        self.train_loss = keras.metrics.Mean("train_loss", dtype=tf.float32)
        self.train_image_cls_loss = keras.metrics.Mean("train_image_cls_loss", dtype=tf.float32)
        self.train_domain_cls_loss = keras.metrics.Mean("train_domain_cls_loss", dtype=tf.float32)
        self.train_image_cls_acc = keras.metrics.Mean("train_image_cls_acc", dtype=tf.float32)
        self.train_domain_cls_acc = keras.metrics.Mean("train_domain_cls_acc", dtype=tf.float32)
        self.val_loss = keras.metrics.Mean("val_loss", dtype=tf.float32)
        self.val_image_cls_loss = keras.metrics.Mean("val_image_cls_loss", dtype=tf.float32)
        self.val_domain_cls_loss = keras.metrics.Mean("val_domain_cls_loss", dtype=tf.float32)
        self.val_image_cls_acc = keras.metrics.Mean("val_image_cls_acc", dtype=tf.float32)
        self.val_domain_cls_acc = keras.metrics.Mean("val_domain_cls_acc", dtype=tf.float32)

    def build_DANN(self):
        """
        这是搭建域适配网络的函数
        :return:
        """
        # 定义源域、目标域的图像输入和DANN模型图像输入
        self.image_input = keras.layers.Input(shape=self.cfg.image_input_shape, name="image_input")

        # 定义域分类器与图像分类器的共享特征
        self.image_feature = build_featur_extractor(self.image_input)
        self.feature_encoder = keras.models.Model()
        # 定义
        # 定义域分类器与图像分类器输出
        self.image_cls_pred = build_image_classify_model(self.image_feature)
        self.domain_cls_pred = build_domain_classify_model(self.image_feature)

        # 定义DANN模型
        self.dann_model = keras.models.Model(self.image_input, [self.image_cls_pred, self.domain_cls_pred])
        self.dann_model.summary()

    def train(self, train_source_datagen, train_target_datagen,val_target_datagen, train_iter_num, val_iter_num):
        """
        这是DANN的训练函数
        :param train_source_datagen: 源域训练数据集生成器
        :param train_target_datagen: 目标域训练数据集生成器
        :param val_datagen: 验证数据集生成器
        :param interval: 验证间隔
        :param train_iter_num: 每个epoch的训练次数
        :param val_iter_num: 每次验证过程的验证次数
        :param pre_model_path: 预训练模型地址,与训练模型为ckpt文件，注意文件路径只需到.ckpt即可。
        """
        # 初始化模型和日志目录路径
        time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        checkpoint_dir = os.path.join(self.cfg.checkpoints_dir, time)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        log_dir = os.path.join(self.cfg.logs_dir, time)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 保存本次训练过程中的参数
        self.cfg.save_config(time)

        # 定义tensorboard相关变量
        self.writer = tf.summary.create_file_writer(log_dir)

        # 定义指数衰减学习率与优化器
        self.exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(
                                initial_learning_rate=self.cfg.init_learning_rate,
                                decay_steps=self.cfg.decay_steps, decay_rate=self.cfg.decay_rate)
        #self.optimizer = tf.keras.optimizers.Adam(self.exponential_decay)
        self.optimizer = tf.keras.optimizers.Adam()

        # 定义最佳验证损失与验证图像分类精度
        best_val_loss = np.Inf
        best_val_image_cls_acc = 0.0

        print('\n----------- start to train -----------\n')
        for ep in np.arange(1, self.cfg.epoch + 1, 1):
            # 进行一个周期的模型训练
            train_loss, train_image_cls_loss, train_domain_cls_loss, train_image_cls_acc, train_domain_cls_acc = \
                self.fit_one_epoch(train_source_datagen, train_target_datagen, train_iter_num, ep)
            # 进行一个周期的模型验证
            val_loss, val_image_cls_loss, val_domain_cls_loss, val_image_cls_acc, val_domain_cls_acc = \
                self.eval_one_epoch(val_target_datagen, val_iter_num, ep)
            # 损失和指标清零
            self.train_loss.reset_states()
            self.train_image_cls_acc.reset_states()
            self.train_domain_cls_loss.reset_states()
            self.train_image_cls_acc.reset_states()
            self.train_domain_cls_acc.reset_states()
            self.val_loss.reset_states()
            self.val_image_cls_acc.reset_states()
            self.val_domain_cls_loss.reset_states()
            self.val_image_cls_acc.reset_states()
            self.val_domain_cls_acc.reset_states()

            # 打印一次训练后的模型训练和验证解雇
            str = "Epoch{:03d}-train_loss-{:.3f}-val_loss-{:.3f}-train_imgae_cls_acc-{:.3f}-val_image_cls_acc-{:.3f}" \
                  "-train_domain_cls_acc-{:.3f}-val_domain_cls_acc-{:.3f}".format \
                (ep, train_loss, val_loss, train_image_cls_acc, val_image_cls_acc, train_domain_cls_acc,
                 val_domain_cls_acc)
            print(str)

            # 保存训练过程中最好的模型
            if val_loss <= best_val_loss or val_image_cls_acc >= best_val_image_cls_acc:
                self.dann_model.save(os.path.join(checkpoint_dir, str + ".h5"))
                if val_loss <= best_val_loss:
                    best_val_loss = val_loss
                if val_image_cls_acc >= best_val_image_cls_acc:
                    best_val_image_cls_acc = val_image_cls_acc
        self.dann_model.save(os.path.join(checkpoint_dir, "trained_dann_mnist.h5"))
        print('\n----------- end to train -----------\n')

    def fit_one_epoch(self, train_source_datagen, train_target_datagen, train_iter_num, ep):
        """
        这是一个周期模型训练的函数
        :param train_source_datagen: 源域训练数据集生成器
        :param train_target_datagen: 目标域训练数据集生成器
        :param train_iter_num: 一个训练周期的迭代次数
        :param ep: 当前训练周期
        :return:
        """
        # 初始化精度条
        progbar = keras.utils.Progbar(train_iter_num)
        print('Epoch {}/{}'.format(ep, self.cfg.epoch))
        for i in np.arange(1, train_iter_num + 1):
            # 更新学习率并可视化
            iter = (ep-1)*train_iter_num+i
            process = iter * 1.0 / (self.cfg.epoch*train_iter_num)
            learning_rate = learning_rate_schedule(process,init_learning_rate=self.cfg.init_learning_rate)
            self.grl_lambd = grl_lambda_schedule(process)
            tf.keras.backend.set_value(self.optimizer.lr,learning_rate)
            with self.writer.as_default():
                tf.summary.scalar("learning_rate",tf.convert_to_tensor(learning_rate),iter)

            # 获取小批量数据集及其图像标签与域标签
            batch_mnist_image_data, batch_mnist_labels = train_source_datagen.__next__()  # train_source_datagen.next_batch()
            batch_mnist_m_image_data, batch_mnist_m_labels = train_target_datagen.__next__()  # train_target_datagen.next_batch()
            batch_source_domain_labels = np.tile([1., 0.], [len(batch_mnist_labels), 1]).astype(np.float32)
            batch_target_domain_labels = np.tile([0., 1.], [len(batch_mnist_m_labels), 1]).astype(np.float32)
            batch_domain_labels = np.vstack([np.tile([1., 0.], [len(batch_mnist_labels), 1]),
                                             np.tile([0., 1.], [len(batch_mnist_m_labels), 1])]).astype(np.float32)

            # 计算梯度
            with tf.GradientTape() as tape:
                # 计算预测源域和目标域预测输出
                source_image_cls_pred, source_domain_cls_pred = self.dann_model(batch_mnist_image_data,training=True)
                target_image_cls_pred, target_domain_cls_pred = self.dann_model(batch_mnist_m_image_data,training=True)

                # 计算预测源域和目标域的相关损失
                source_image_cls_loss = keras.losses.categorical_crossentropy(batch_mnist_labels,source_image_cls_pred)
                source_domain_cls_loss = keras.losses.categorical_crossentropy(batch_source_domain_labels,source_domain_cls_pred)
                target_domain_cls_loss = keras.losses.categorical_crossentropy(batch_target_domain_labels,target_domain_cls_pred)

                # 计算域分类损失和模型损失
                #domain_cls_loss = tf.reduce_mean(source_domain_cls_loss + target_domain_cls_loss)
                #loss = tf.reduce_mean(source_image_cls_loss) + domain_cls_loss

                domain_cls_loss = source_domain_cls_loss + target_domain_cls_loss
                loss = source_image_cls_loss + domain_cls_loss

                # 计算图像分类精度和域分类精度
                image_cls_acc = keras.metrics.categorical_accuracy(batch_mnist_labels, source_image_cls_pred)
                domain_cls_acc = keras.metrics.categorical_accuracy \
                    (batch_domain_labels, tf.concat([source_domain_cls_pred, target_domain_cls_pred], axis=0))
            # 自定义优化过程
            vars = tape.watched_variables()
            grads = tape.gradient(loss, vars)
            self.optimizer.apply_gradients(zip(grads, vars))

            # 更新进度条
            progbar.update(i, [('loss', loss),
                               ('image_cls_loss', source_image_cls_loss),
                               ('domain_cls_loss', domain_cls_loss),
                               ("image_acc", image_cls_acc),
                               ("domain_acc", domain_cls_acc)])

            # 计算平均损失与精度
            self.train_loss(loss)
            self.train_image_cls_loss(source_image_cls_loss)
            self.train_domain_cls_loss(domain_cls_loss)
            self.train_image_cls_acc(image_cls_acc)
            self.train_domain_cls_acc(domain_cls_acc)

        # 可视化损失与指标
        with self.writer.as_default():
            tf.summary.scalar("train/loss", self.train_loss.result(), ep)
            tf.summary.scalar("train/image_cls_loss", self.train_image_cls_loss.result(), ep)
            tf.summary.scalar("train/domain_cls_loss", self.train_domain_cls_loss.result(), ep)
            tf.summary.scalar("train/image_acc", self.train_image_cls_acc.result(), ep)
            tf.summary.scalar("train/domain_acc", self.train_domain_cls_acc.result(), ep)

        return self.train_loss.result(), self.train_image_cls_loss.result(), self.train_domain_cls_loss.result(), \
               self.train_image_cls_acc.result(), self.train_domain_cls_acc.result()

    def eval_one_epoch(self, val_target_datagen, val_iter_num, ep):
        """
        这是一个周期的模型验证函数
        :param val_target_datagen: 目标域验证数据集生成器
        :param val_iter_num: 一个验证周期的迭代次数
        :param ep: 当前验证周期
        :return:
        """
        for i in np.arange(1, val_iter_num + 1):
            # 获取小批量数据集及其图像标签与域标签
            batch_mnist_m_image_data, batch_mnist_m_labels = val_target_datagen.__next__()
            batch_target_domain_labels = np.tile([0., 1.], [len(batch_mnist_m_labels), 1]).astype(np.float32)

            # 计算目标域数据的预测输出
            target_image_cls_pred, target_domain_cls_pred = self.dann_model(batch_mnist_m_image_data,training=False)

            # 计算目标域预测相关损失
            target_image_cls_loss = keras.losses.categorical_crossentropy(target_domain_cls_pred,
                                                                           batch_target_domain_labels)
            target_domain_cls_loss = keras.losses.categorical_crossentropy(target_domain_cls_pred,
                                                                            batch_target_domain_labels)
            '''
            domain_cls_loss = tf.reduce_mean(target_domain_cls_loss)
            image_cls_loss = tf.reduce_mean(target_image_cls_loss)
            loss = image_cls_loss + domain_cls_loss
            '''
            domain_cls_loss = target_domain_cls_loss
            image_cls_loss = target_image_cls_loss
            loss = image_cls_loss + domain_cls_loss

            # 计算目标域图像分类精度与域分类精度
            image_acc = keras.metrics.categorical_accuracy(batch_mnist_m_labels, target_image_cls_pred)
            domain_acc = keras.metrics.categorical_accuracy(batch_target_domain_labels, target_domain_cls_pred)

            # 更新训练损失与训练精度
            self.val_loss(loss)
            self.val_image_cls_loss(image_cls_loss)
            self.val_domain_cls_loss(domain_cls_loss)
            self.val_image_cls_acc(image_acc)
            self.val_domain_cls_acc(domain_acc)

        # 可视化验证损失及其指标
        with self.writer.as_default():
            tf.summary.scalar("val/loss", self.val_loss.result(), ep)
            tf.summary.scalar("val/image_cls_loss", self.val_image_cls_loss.result(), ep)
            tf.summary.scalar("val/domain_cls_loss", self.val_domain_cls_loss.result(), ep)
            tf.summary.scalar("val/image_acc", self.val_image_cls_acc.result(), ep)
            tf.summary.scalar("val/domain_acc", self.val_domain_cls_acc.result(), ep)

        return self.val_loss.result(), self.val_image_cls_loss.result(), self.val_domain_cls_loss.result(), \
               self.val_image_cls_acc.result(), self.val_domain_cls_acc.result()
