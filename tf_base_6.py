# -*- coding: utf-8 -*-
"""
Created on Thu May 17 23:04:31 2018

@author: Administrator
"""
import cv2 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage import color,transform

sess = tf.Session()  
new_saver = tf.train.import_meta_graph('trained.ckpt.meta')  
new_saver.restore(sess, tf.train.latest_checkpoint('./'))  
graph = tf.get_default_graph()  
x = graph.get_tensor_by_name('x:0')  
y_ = graph.get_tensor_by_name('y_:0')  
new_y = graph.get_tensor_by_name('new_y:0')  
keep_prob = graph.get_tensor_by_name('keep_prob:0')  

imgs = []
im = cv2.imread("C:/Users/Administrator/Desktop/qwe/1 (22).jpg")
img = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

image = transform.resize(img,(44,36,1))
imgs.append(image)
    
a = np.array([8])
new = sess.run([new_y],feed_dict = {x:imgs,y_:a,keep_prob:1.0})
position = np.argmax(new)
name = ['薛义权','宋锦涛','蓝高杰','陆权忠','覃绍彬','周俊成','黄凯','潘俊伟','陈东良']
print(new)
print('result ：',name[position]) 
plt.imshow(img,cmap='gray'),plt.show()