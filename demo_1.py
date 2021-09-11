# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 21:34:25 2019

@author: Dufert
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

input_x = np.random.rand(3000, 1)
noise = np.random.uniform(0, 0.05, input_x.shape)
output_y = input_x*8 +1 + noise

weight = tf.Variable(dtype=tf.float32, initial_value=np.random.rand(input_x.shape[1], 1))
bias = tf.Variable(dtype=tf.float32, initial_value=np.random.rand(input_x.shape[1], 1))

x = tf.placeholder(tf.float32, [None, input_x.shape[1]])
y_ = tf.matmul(x, weight) + bias

# 计算loss函数
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_-output_y),1))
train = tf.train.GradientDescentOptimizer(0.25).minimize(loss)

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(train, feed_dict={x: input_x})
    
    print(weight.eval(), bias.eval())
    test = np.mat([[1.], [2.], [3.]])
    print(sess.run(y_, feed_dict={x: test}))
    



