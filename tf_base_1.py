# -*- coding: utf-8 -*-
"""
Created on Wed May 16 11:52:01 2018
# 实现了一个简单的神经网络，只包含一层全连接层，使用softmax计算输出概率分布，
# 对mnist数据集进行训练，不保留结构特征，将数据集转换成向量计算得出92%以上的
# 检测正确率，在后部分训练达到50左右时，可能是出现了梯度耗尽问题，准确率无法
# 继续提升，反而下降到出事水平。
@author: Administrator
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

w = 28;h = 28;c = 1
with tf.device('/cpu:0'):
    
    mnist = input_data.read_data_sets("MINST_data/",one_hot=True)
    
    x = tf.placeholder("float",[None,784])
    y_ = tf.placeholder("float",[None,10])
    
    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x,W)+b)
    
    cross_sh = - tf.reduce_sum(y_*tf.log(y))
    train_op = tf.train.AdamOptimizer(0.001).minimize(cross_sh)
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,1),tf.argmax(y_,1)),"float"))
    
    init = tf.global_variables_initializer()
    
with tf.Session(config= tf.ConfigProto(log_device_placement=True)) as sess:
    A = np.zeros(60)
    B = np.zeros(600)
    C = np.zeros(60)
    sess.run(init)
    for j in range(60):    
        accuracy = 0
        for i in range(600):
            train_data,train_label = mnist.train.next_batch(100)
            _,accu = sess.run([train_op,acc],feed_dict = {x: train_data,y_: train_label})
            accuracy += accu
            if i==0:
                B[i] = accu
        A[j] = accuracy/600
        C[j] = sess.run(acc,feed_dict = {x:mnist.test.images,y_:mnist.test.labels})
        print('第',str(j),'次 tr_acc: ',A[j])    
        print('第',str(j),'次 te_acc: ',C[j])
    plt.plot(A),plt.plot(C)

    
