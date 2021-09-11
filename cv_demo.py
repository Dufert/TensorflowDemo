# -*- coding: utf-8 -*-
"""
Created on Sun May 13 19:38:55 2018

@author: Administrator
"""
import cv2 
import numpy as np
import tensorflow as tf
import os,glob
from skimage import io,color,transform

w = 28
h = 28
c = 1
suffix = '/*.png'
train_path = 'g:/Computer Vision/Library/mnist/train/'
test_path = 'g:/Computer Vision/Library/mnist/test/'
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