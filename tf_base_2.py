# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 14:03:17 2018
# 较base_1，此处实现了一个简单的卷积神经网络，结构为 2*（conv、relu、pool）
# fc、softmax组成。学习率为0.001，训练集合为 mnist，在卷积层使用了全零填充以保留边
# 缘特性在训练阶段加入了dropout，随机置零权值，预防过拟合，其在训练时，网络耗时过长，
# 考虑需要使用变换的学习率进行改善。
@author: Administrator
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

w = 28;h = 28;c = 1
mnist = input_data.read_data_sets("MINST_data/",one_hot=True)

x = tf.placeholder("float",[None,784])
y_ = tf.placeholder("float",[None,10])

X = tf.reshape(x,[-1,w,h,c])#-1表示我懒得计算该填什么数字

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")

def relu(x,bias):
    return tf.nn.relu(x+bias)

def max_pool_2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

conv_weight_1 = weight_variable([5,5,1,6])
conv_bias_1 = bias_variable([6])

conv_conv_1 = conv2d(X,conv_weight_1)
conv_relu_1 = relu(conv_conv_1,conv_bias_1)
conv_pool_1 = max_pool_2(conv_relu_1)


conv_weight_2 = weight_variable([5,5,6,16])
conv_bias_2 = bias_variable([16])

conv_conv_2 = conv2d(conv_pool_1,conv_weight_2)
conv_relu_2 = relu(conv_conv_2,conv_bias_2)
conv_pool_2 = max_pool_2(conv_relu_2)

#现在为7*7*16数据
nodes = 7*7*16 
reshaped = tf.reshape(conv_pool_2,[-1,nodes])

fc_weight_1 = weight_variable([nodes,120])
fc_bias_1 = bias_variable([120])
fc_out_1 = tf.nn.relu(tf.matmul(reshaped,fc_weight_1)+fc_bias_1)

keep_prob = tf.placeholder("float")
fc_drop_1 = tf.nn.dropout(fc_out_1, keep_prob)

fc_weight_2 = weight_variable([120,10])
fc_bias_2 = bias_variable([10])
y = tf.nn.softmax(tf.matmul(fc_drop_1,fc_weight_2)+fc_bias_2)

cross_sh = -tf.reduce_sum(tf.log(y+1e-10)*y_)
#cross_sh = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=y_)
train_op = tf.train.AdamOptimizer(0.001).minimize(cross_sh)
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
acc = tf.reduce_mean(tf.cast(correct_prediction,"float"))

initial = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(initial)
    train_num = 60
    A = np.zeros(train_num)
    B = np.zeros(train_num)
    for j in range(train_num):    
        accuracy = 0
        for i in range(600):
            train_data,train_label = mnist.train.next_batch(100)
            _,accu = sess.run([train_op,acc],feed_dict = {
                    x: train_data,
                    y_: train_label,
                    keep_prob: 0.5})
            accuracy += accu
        A[j] = accuracy/600
        print('第',str(j),'次 tra_acc: ',A[j])
        B[j] = sess.run(acc,feed_dict = {
                x:mnist.test.images,
                y_:mnist.test.labels,
                keep_prob: 1.0})
        print('第',str(j),'次 tes_acc: ',B[j])
