# -*- coding: utf-8 -*-
"""
Created on Wed May 16 12:09:27 2018
# 采用结构简单的卷积神经网络，网络结构为 conv、relu、pool、fc、softmax 使用mnist集合
# 进行试验，对数据集进行了随机排序调整，代替原本有序排列的数据，分每1000张为一批，
# 在初始阶段时正确率得到了有效提升，同时对网络添加了L2正则化步骤，以防止过拟合，但取消了
# dropout层，网络收敛速度得到提升，
@author: Administrator
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from cv_demo import train_data,train_label
from cv_demo import test_data,test_label

w = 28;h = 28;c = 1

tf.reset_default_graph()
def inference(input_tensor,regularizer,c):
    
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable('weight',[5,5,c,6],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable('bias',[6],initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,1,1,1],padding='VALID')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))
        
    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        
    pool_shape = pool1.get_shape().as_list()
    nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
    reshaped = tf.reshape(pool1,[-1,nodes])
 
    with tf.variable_scope('layer3-fc1'):
        fc3_weights = tf.get_variable('weight',[nodes,10],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc3_weights))
        fc3_biases = tf.get_variable('bias',[10],initializer=tf.truncated_normal_initializer(stddev=0.1))
        logit = tf.matmul(reshaped,fc3_weights) + fc3_biases
    return logit

def get_batch(data,label,batch_size):
    for start_index in range(0,len(data)-batch_size+1,batch_size):
        slice_index = slice(start_index,start_index+batch_size)
        yield data[slice_index],label[slice_index]


x = tf.placeholder(tf.float32,[None,w,h,c],name='x')#运行时候必须在session会话中输入数据
y_ = tf.placeholder(tf.int32,[None],name='y_')

regularizer = tf.contrib.layers.l2_regularizer(0.001)#L2正则化 减少网络结构 防止过拟合
y = inference(x,regularizer,c) 
new_y = tf.nn.softmax(y,name = 'new_y') 
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=y_) 
cross_entropy_mean = tf.reduce_mean(cross_entropy)#如果求loss，则要做一步tf.reduce_mean操作，对向量求均值！
loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))#tf.get_collection：从一个结合中取出全部变量，是一个列表
train_op = tf.train.AdamOptimizer(0.001).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(y,1),tf.int32),y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    batch_size = 1000
    train_num = 10
    Tra_acc = np.zeros(train_num)
    Tes_acc = np.zeros(train_num)
    for i in range(train_num):
        train_loss,train_acc,batch_num = 0, 0, 0
        for train_data_batch,train_label_batch in get_batch(train_data,train_label,batch_size):
            _,tra_err,tra_acc = sess.run([train_op,loss,accuracy],feed_dict={
                    x:train_data_batch,
                    y_:train_label_batch})
            train_loss+=tra_err
            train_acc+=tra_acc
            batch_num+=1
        Tra_acc[i] = train_acc/batch_num
        print('第',str(i),'次 train acc:',Tra_acc[i])
        
        test_acc,batch_num_t = 0, 0
        for test_data_batch,test_label_batch in get_batch(test_data,test_label,batch_size):
            tes_acc = sess.run(accuracy,feed_dict={
                    x:test_data_batch,
                    y_:test_label_batch})
            test_acc+=tes_acc
            batch_num_t+=1
        Tes_acc[i] = test_acc/batch_num_t
        print('第',str(i),'次 test acc:',Tes_acc[i])
        plt.plot(Tra_acc),plt.plot(Tes_acc)
        
"""
一组结果呈现：
        第 0 次 train acc: 0.47120000310242177
        第 0 次 test acc: 0.7681000053882598
        第 1 次 train acc: 0.8241333335638046
        第 1 次 test acc: 0.8829000055789947
        第 2 次 train acc: 0.8852499991655349
        第 2 次 test acc: 0.91010000705719
        第 3 次 train acc: 0.9049833347400029
        第 3 次 test acc: 0.9200000047683716
        第 4 次 train acc: 0.9167666693528493
        第 4 次 test acc: 0.9288000106811524
        第 5 次 train acc: 0.9249833305676778
        第 5 次 test acc: 0.9367999970912934
"""