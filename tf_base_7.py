# -*- coding: utf-8 -*-
"""
Created on Sat May 26 15:19:45 2018

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 16 16:26:03 2018
# 对于梯度耗尽的问题 进行改善
@author: Administrator
"""
import cv2 
import os,glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage import color,transform

tf.reset_default_graph()

w = 44
h = 36
c = 1
suffix = '/*.bmp'

learning_rate = 0.03  # 初始学习速率时0.1
decay_rate = 0.5  # 衰减率
#global_steps = 1000  # 总的迭代次数
decay_steps = 100  # 衰减次数
test_path = "E:/BaiduYunDownload/ORL/ORL_test/"
train_path = "E:/BaiduYunDownload/ORL/ORL_train/"
#test_path = "E:/BaiduYunDownload/mylibrary/test/"
#train_path = "E:/BaiduYunDownload/mylibrary/train/"

def read_image(path,w,h,c):
    
    label_dir = [path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    images = []
    labels = []
    for index,folder in enumerate(label_dir):
        for img in glob.glob(folder+suffix):
            image = cv2.imread(img)
            image = color.rgb2gray(image)
            image = transform.resize(image,(w,h,c))
            images.append(image)
            labels.append(index)
    return np.asarray(images,dtype=np.float32),np.asarray(labels,dtype=np.int32)
        
train_data,train_label = read_image(train_path,w,h,c)
train_image_num = len(train_data)
train_image_index = np.arange(train_image_num)
np.random.shuffle(train_image_index)
train_data = train_data[train_image_index]
train_label = train_label[train_image_index]

test_data,test_label = read_image(test_path,w,h,c)
test_image_num = len(test_data)
test_image_index = np.arange(test_image_num)
np.random.shuffle(test_image_index)
test_data = test_data[test_image_index]
test_label = test_label[test_image_index]

def inference(input_tensor,regularizer,keep_prob):
    
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable('weight',[5,5,1,16],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable('bias',[16],initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,1,1,1],padding='VALID')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))
        
    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        
    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.get_variable('weight',[5,5,16,32],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable('bias',[32],initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1,conv2_weights,strides=[1,1,1,1],padding='VALID')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))
        
    with tf.name_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')  
        
    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
    reshaped = tf.reshape(pool2,[-1,nodes])
 
    
    
    
    
    with tf.variable_scope('layer5-fc1'):
        fc3_weights = tf.get_variable('weight',[nodes,120],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc3_weights))
        fc3_biases = tf.get_variable('bias',[120],initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc3_out = tf.matmul(reshaped,fc3_weights) + fc3_biases
        
    dropout_out = tf.nn.dropout(fc3_out, keep_prob)
        
    with tf.variable_scope('layer6-fc2'):
        fc4_weights = tf.get_variable('weight',[120,21],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc4_weights))
        fc4_biases = tf.get_variable('bias',[21],initializer=tf.truncated_normal_initializer(stddev=0.1))
        out = tf.matmul(dropout_out,fc4_weights) + fc4_biases
        
    return out

def get_batch(data,label,batch_size):
    for start_index in range(0,len(data)-batch_size+1,batch_size):
        slice_index = slice(start_index,start_index+batch_size)
        yield data[slice_index],label[slice_index]

keep_prob = tf.placeholder("float",name='keep_prob')
x = tf.placeholder(tf.float32,[None,w,h,c],name='x') 
y_ = tf.placeholder(tf.int32,[None],name='y_')

regularizer = tf.contrib.layers.l2_regularizer(0.001)#L2正则化 减少网络结构 防止过拟合
y = inference(x,regularizer,keep_prob) 

new_y = tf.nn.softmax(y,name='new_y')

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=y_) 
cross_entropy_mean = tf.reduce_mean(cross_entropy) 
#tf.get_collection：从一个结合中取出全部变量，是一个列表
loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

global_steps = tf.Variable(tf.constant(0))

learning_rate = tf.train.exponential_decay(0.03,  
                                           global_steps,  
                                           decay_steps,  
                                           decay_rate,
                                           staircase=True)
#如果staircase=True，那就表明每decay_steps次计算学习速率变化，更新原始学习速率，如果是False，那就是每一步都更新学习速率。红色表示False，绿色表示True。
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)#tf.train.AdamOptimizer(learning_rate).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(y,1),tf.int32),y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    batch_size = 10
    train_num = 150
    rate = 0.001
    Tra_acc = np.zeros(train_num)
    Tes_acc = np.zeros(train_num)
    for i in range(train_num):
        train_loss,train_acc,batch_num = 0, 0, 0
        for train_data_batch,train_label_batch in get_batch(train_data,train_label,batch_size):
            _,tra_err,tra_acc,learn = sess.run([train_op,loss,accuracy,learning_rate],feed_dict={
                    x:train_data_batch,
                    y_:train_label_batch,
                    keep_prob:0.5,
                    global_steps:i})
            train_loss+=tra_err
            train_acc+=tra_acc
            batch_num+=1
        
        Tra_acc[i] = train_acc/batch_num
#        if (Tra_acc[i]<0.75) & (Tra_acc[i]>0.5):
#            rate = 0.001
#        elif Tra_acc[i]>0.75:
#            rate = 0.001
        print('第',str(i),'次 train acc:',Tra_acc[i])
        
        test_acc,batch_num_t = 0, 0
        for test_data_batch,test_label_batch in get_batch(test_data,test_label,batch_size):
            tes_acc = sess.run(accuracy,feed_dict={
                    x:test_data_batch,
                    y_:test_label_batch,
                    keep_prob:1.0,global_steps:i})
            test_acc+=tes_acc
            batch_num_t+=1
        Tes_acc[i] = test_acc/batch_num_t
        print('第',str(i),'次 test acc:',Tes_acc[i])
    plt.plot(Tra_acc),plt.plot(Tes_acc)
#    saver.save(sess,"./trained.ckpt")