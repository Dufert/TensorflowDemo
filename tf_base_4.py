# -*- coding: utf-8 -*-
"""
Created on Wed May 16 14:42:35 2018
#  写一个比较完善的CNN网络，同时包含L2正则化、dropout层、可变学习率、其结构为
# （conv、relu、max_pool、fc1、dropout、fc2、softmax）在前期可以获得较快的收敛速度
#  由于自行设置的可变学习率函数较为低级 无法适应所有情况 
@author: Administrator
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from cv_demo import train_data,train_label
from cv_demo import test_data,test_label

w = 28;h = 28;c = 1

tf.reset_default_graph()
def inference(input_tensor,regularizer,keep_prob):
    
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable('weight',[5,5,1,6],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable('bias',[6],initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,1,1,1],padding='VALID')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))
        
    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        
    pool_shape = pool1.get_shape().as_list()
    nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
    reshaped = tf.reshape(pool1,[-1,nodes])
 
    with tf.variable_scope('layer3-fc1'):
        fc3_weights = tf.get_variable('weight',[nodes,120],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc3_weights))
        fc3_biases = tf.get_variable('bias',[120],initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc3_out = tf.matmul(reshaped,fc3_weights) + fc3_biases
        
    dropout_out = tf.nn.dropout(fc3_out, keep_prob)
        
    with tf.variable_scope('layer4-fc2'):
        fc4_weights = tf.get_variable('weight',[120,10],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc4_weights))
        fc4_biases = tf.get_variable('bias',[10],initializer=tf.truncated_normal_initializer(stddev=0.1))
        out = tf.matmul(dropout_out,fc4_weights) + fc4_biases
        
    return out

def get_batch(data,label,batch_size):
    for start_index in range(0,len(data)-batch_size+1,batch_size):
        slice_index = slice(start_index,start_index+batch_size)
        yield data[slice_index],label[slice_index]

keep_prob = tf.placeholder("float")
x = tf.placeholder(tf.float32,[None,w,h,c],name='x') 
y_ = tf.placeholder(tf.int32,[None],name='y_')

regularizer = tf.contrib.layers.l2_regularizer(0.001)#L2正则化 减少网络结构 防止过拟合
y = inference(x,regularizer,keep_prob) 

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=y_) 
cross_entropy_mean = tf.reduce_mean(cross_entropy) 
#tf.get_collection：从一个结合中取出全部变量，是一个列表
loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

learn_rate = tf.placeholder("float")

train_op = tf.train.AdamOptimizer(learn_rate).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(y,1),tf.int32),y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    batch_size = 1000
    train_num = 10
    rate = 0.01
    Tra_acc = np.zeros(train_num)
    Tes_acc = np.zeros(train_num)
    for i in range(train_num):
        train_loss,train_acc,batch_num = 0, 0, 0
        for train_data_batch,train_label_batch in get_batch(train_data,train_label,batch_size):
            _,tra_err,tra_acc = sess.run([train_op,loss,accuracy],feed_dict={
                    x:train_data_batch,
                    y_:train_label_batch,
                    learn_rate:rate,
                    keep_prob:0.5})
            train_loss+=tra_err
            train_acc+=tra_acc
            batch_num+=1
        Tra_acc[i] = train_acc/batch_num
        if (Tra_acc[i]<0.75) & (Tra_acc[i]>0.5):
            rate = 0.001
        elif Tra_acc[i]>0.75:
            rate = 0.001
        print('第',str(i),'次 train acc:',Tra_acc[i])
        
        test_acc,batch_num_t = 0, 0
        for test_data_batch,test_label_batch in get_batch(test_data,test_label,batch_size):
            tes_acc = sess.run(accuracy,feed_dict={
                    x:test_data_batch,
                    y_:test_label_batch,
                    learn_rate:0.001,
                    keep_prob:1.0})
            test_acc+=tes_acc
            batch_num_t+=1
        Tes_acc[i] = test_acc/batch_num_t
        print('第',str(i),'次 test acc:',Tes_acc[i])
    plt.plot(Tra_acc),plt.plot(Tes_acc)
"""
    实验结果：
        第 0 次 train acc: 0.8591166689991951
        第 0 次 test acc: 0.9675999879837036
        第 1 次 train acc: 0.9592166682084401
        第 1 次 test acc: 0.97289999127388
        第 2 次 train acc: 0.9638000011444092
        第 2 次 test acc: 0.9755999922752381
        第 3 次 train acc: 0.9660833319028218
        第 3 次 test acc: 0.9770000040531158
        第 4 次 train acc: 0.9679999987284342
        第 4 次 test acc: 0.9782000005245208
        第 5 次 train acc: 0.9698666671911875
        第 5 次 test acc: 0.9790000081062317
        第 6 次 train acc: 0.9714333325624466
        第 6 次 test acc: 0.9796000003814698
        第 7 次 train acc: 0.9734666695197424
        第 7 次 test acc: 0.9796999931335449
        第 8 次 train acc: 0.9734500020742416
        第 8 次 test acc: 0.980299997329712
        第 9 次 train acc: 0.9751333326101304
        第 9 次 test acc: 0.9814000010490418
"""