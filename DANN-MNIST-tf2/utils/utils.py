# -*- coding: utf-8 -*-
# @Time    : 2020/7/10 15:12
# @Author  : DaiPuwei
# @Email   : 771830171@qq.com
# @File    : utils.py
# @Software: PyCharm

import numpy as np

def learning_rate_schedule(process,init_learning_rate = 0.01,alpha = 10.0 , beta = 0.75):
    """
    这个学习率的变换函数
    :param process: 训练进程比率，值在0-1之间
    :param init_learning_rate: 初始学习率，默认为0.01
    :param alpha: 参数alpha，默认为10
    :param beta: 参数beta，默认为0.75
    """
    return init_learning_rate /(1.0 + alpha * process)**beta

def grl_lambda_schedule(process,gamma=10.0):
    """
    这是GRL的参数lambda的变换函数
    :param process: 训练进程比率，值在0-1之间
    :param gamma: 参数gamma，默认为10
    """
    return 2.0 / (1.0+np.exp(-gamma*process)) - 1.0