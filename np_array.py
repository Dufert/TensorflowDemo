# -*- coding: utf-8 -*-
"""
Created on Mon May 14 12:20:35 2018
# Do easy workouts
@author: Administrator
"""

import numpy as np
import tensorflow as tf

a = np.array([1,2,3])
b = np.array([[1,2,3]])
c = np.array([[1,2,3],[1,2,3]])
print(a,b)
print(c)
d = (-1,2,3)
print(d[1])
e = np.zeros(4)
f = np.zeros((4))
print(e,f)

aa = np.linspace(-1,1,300)[:,np.newaxis]#增加维度
yy = np.square(c) #单个元素的平方
print(yy)

y = tf.placeholder(tf.float32,[1],name='y')

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(y[0],feed_dict={y:[1]}))

rate = 0.1
Tra_acc = np.zeros(10)
for i in range(10):
    if (Tra_acc[i]< 0.75) & (Tra_acc[i]> np.float64(0.5)):
        rate = 0.01
