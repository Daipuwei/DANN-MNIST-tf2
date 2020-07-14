# -*- coding: utf-8 -*-
# @Time    : 2020/2/15 16:36
# @Author  : Dai PuWei
# @Email   : 771830171@qq.com
# @File    : train.py
# @Software: PyCharm

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'
import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow import keras

from config.config import config
from model.MNIST2MNIST_M import MNIST2MNIST_M_DANN
#from model.dann_train import MNIST2MNIST_M_DANN

#tf.enable_eager_execution()
#tfe = tf.contrib.eager

def shuffle_aligned_list(data):
    """Shuffle arrays in a list by shuffling each array identically."""
    num = data[0].shape[0]
    p = np.random.permutation(num)
    return [d[p] for d in data]

def batch_generator(data, batch_size, shuffle=True):
    """Generate batches of data.

    Given a list of array-like objects, generate batches of a given
    size by yielding a list of array-like objects corresponding to the
    same slice of each input.
    """
    if shuffle:
        data = shuffle_aligned_list(data)

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= len(data[0]):
            batch_count = 0

            if shuffle:
                data = shuffle_aligned_list(data)

        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data]

def run_main():
    """
       这是主函数
    """
    # 初始化参数配置类
    cfg = config()

    # Load MNIST
    (mnist_x_train, mnist_y_train), (mnist_x_test, mnist_y_test) = keras.datasets.mnist.load_data()
    mnist_x_train = np.expand_dims(mnist_x_train,-1)
    mnist_x_test = np.expand_dims(mnist_x_test,-1)
    mnist_x_train = np.concatenate([mnist_x_train, mnist_x_train, mnist_x_train], 3).astype(np.float32)
    mnist_x_test = np.concatenate([mnist_x_test, mnist_x_test, mnist_x_test], 3).astype(np.float32)
    mnist_y_train = keras.utils.to_categorical(mnist_y_train).astype(np.float32)
    mnist_y_test = keras.utils.to_categorical(mnist_y_test).astype(np.float32)

    # Load MNIST-M
    mnistm = pkl.load(open(os.path.abspath('./dataset/mnistm/mnistm_data.pkl'), 'rb'))
    mnistm_train = mnistm['train'].astype(np.float32)
    mnistm_valid = mnistm['valid'].astype(np.float32)

    # Compute pixel mean for normalizing data
    pixel_mean = np.vstack([mnist_x_train, mnistm_train]).mean((0, 1, 2))
    cfg.set(pixel_mean=pixel_mean)

    mnist_x_train = (mnist_x_train - pixel_mean) / 255.0
    mnistm_train = (mnistm_train - pixel_mean)/ 255.0
    mnist_x_test = (mnist_x_test - pixel_mean)/ 255.0
    mnistm_valid = (mnistm_valid - pixel_mean)/ 255.0

    # 构造数据生成器
    train_source_datagen = batch_generator([mnist_x_train,mnist_y_train],cfg.batch_size // 2)
    train_target_datagen = batch_generator([mnistm_train,mnist_y_train],cfg.batch_size // 2)
    val_target_datagen = batch_generator([mnistm_valid,mnist_y_test],cfg.batch_size)

    """
    train_source_datagen = DataGenerator(os.path.join(cfg.dataset_dir, 'mnist'),int(cfg.batch_size/2),
                                         cfg.image_size,source_flag=True,mode="train")
    train_target_datagen = DataGenerator(os.path.join(cfg.dataset_dir, 'mnistM'),int(cfg.batch_size/2),
                                         cfg.image_size,source_flag=False,mode="train")
    val_datagen = DataGenerator(os.path.join(cfg.dataset_dir, 'mnistM'),cfg.batch_size,
                                cfg.image_size,source_flag=False,mode="val")
    """

    # 初始化每个epoch的训练次数和每次验证过程的验证次数
    train_source_batch_num = int(len(mnist_x_train) // (cfg.batch_size // 2))
    train_target_batch_num = int(len(mnistm_train) // (cfg.batch_size // 2))
    train_iter_num = int(np.max([train_source_batch_num,train_target_batch_num]))

    val_iter_num = int(len(mnistm_valid) // cfg.batch_size)

    # 初始化DANN，并进行训练
    dann = MNIST2MNIST_M_DANN(cfg)
    #pre_model_path = os.path.abspath("./pre_model/trained_model.ckpt")
    dann.train(train_source_datagen,train_target_datagen,val_target_datagen,train_iter_num,val_iter_num)

if __name__ == '__main__':
    run_main()