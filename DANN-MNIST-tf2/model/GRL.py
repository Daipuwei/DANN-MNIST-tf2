# -*- coding: utf-8 -*-
# @Time    : 2020/2/14 20:59
# @Author  : Dai PuWei
# @Email   : 771830171@qq.com
# @File    : GRL.py
# @Software: PyCharm

import tensorflow as tf

@tf.custom_gradient
def gradient_reversal(x,alpha=1.0):
	def grad(dy):
		return -dy * alpha, None
	return x, grad
