# -*- coding: utf-8 -*-
"""
Created on Sat May 26 00:12:50 2018

@author: Administrator
"""

1# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 14:42:06 2018

@author: Administrator
"""
import cv2
import matplotlib.pyplot as pplt
import numpy as np
from skimage import transform
import tensorflow as tf
import matplotlib.image as plb
import matplotlib.pyplot as plt

 
classifier_path = "H:/Anaconda/Anaconda_exercise/RLJC/haarcascade_frontalface_default.xml"
#sess = tf.Session()  
#new_saver = tf.train.import_meta_graph('trained_variables.ckpt.meta')  
#new_saver.restore(sess, tf.train.latest_checkpoint('./'))  
#graph = tf.get_default_graph()  
#x = graph.get_tensor_by_name('x:0')  
#y_ = graph.get_tensor_by_name('y_:0')  
#new_y = graph.get_tensor_by_name('new_y:0')
#keep_prob = graph.get_tensor_by_name('keep_prob:0')  
xc = 1
if xc == 1:
    img_1=plb.imread("C:/Users/Administrator/Desktop/face/kuang.jpg")
    img_1 = np.dot(img_1[...,:3], [0.299, 0.587, 0.114])
    img_raw,img_col = np.shape(img_1)
    imgs = []
    classfier = cv2.CascadeClassifier(classifier_path)
    color = (0, 255, 0)
    frame=plb.imread("C:/Users/Administrator/Desktop/qwe/2 (1).jpg")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                 
    faceRects = classfier.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(30, 30),
                flags = 0)
    if len(faceRects) > 0:              #大于0则检测到人脸                                   
        for faceRect in faceRects:      #单独框出每一张人脸
            x, y, w, h = faceRect        
    image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
    plt.imshow(gray,cmap='gray'),plt.show()
    image = np.dot(image[...,:3], [0.299, 0.587, 0.114])
    img_2 = transform.resize(image,(100,100))
#    for i in range(img_raw):
#        for j in range(img_col):
#            if img_1[i][j] == 0:
#                img_2[i,j]=0

    new_img = np.zeros((img_raw-11,img_col-27))
    new_img_raw,new_img_col = np.shape(new_img)
    for i in range(new_img_raw):
        for j in range(new_img_col):
            new_img[i][j] = img_2[i+3][j+13]
    images = transform.resize(new_img,(88,72,1))
    plt.imshow(new_img,cmap='gray'),plt.show()
    cv2.imwrite("C:/Users/Administrator/Desktop/qwe/2 (12).jpg",new_img)
#    imgs.append(images)
    
#    faceCascade = cv2.CascadeClassifier(classifier_path)
#    imgs = []
#    img_1=plb.imread("C:/Users/Administrator/Desktop/face/kuang.jpg")
#    img_1 = np.dot(img_1[...,:3], [0.299, 0.587, 0.114])
#    img_raw,img_col = np.shape(img_1)
#    im = plb.imread("C:/Users/Administrator/Desktop/qwe/2 (1).jpg")
#    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)    
#    faces = faceCascade.detectMultiScale(
#        gray,
#        scaleFactor=1.2,
#        minNeighbors=5,
#        minSize=(30, 30),
#        flags = 0)
#    x = faces[0,0]-10
#    y = faces[0,1]-10
#    w = faces[0,2]+10
#    h = faces[0,3]+10
#    image = gray[y:(y+h),x:(x+w)]
#    plt.imshow(image,cmap='gray'),plt.show()
#    img_2 = transform.resize(image,(100,100))
#    for i in range(img_raw):
#        for j in range(img_col):
#            if img_1[i][j] == 0:
#                img_2[i,j]=0
#    plt.imshow(img_2,cmap='gray'),plt.show()
#    new_img = np.zeros((img_raw-11,img_col-27))
#    new_img_raw,new_img_col = np.shape(new_img)
#    for i in range(new_img_raw):
#        for j in range(new_img_col):
#            new_img[i][j] = img_2[i+3][j+13]
#    plt.imshow(new_img,cmap='gray'),plt.show()
#    images = transform.resize(new_img,(44,36,1))
#    images= np.float32(images)
#    imgs.append(images)
    
else:
    def catch_video(window_name, camera_idx):
        
        img_1=plb.imread("C:/Users/Administrator/Desktop/face/kuang.jpg")
        img_1 = np.dot(img_1[...,:3], [0.299, 0.587, 0.114])
        img_raw,img_col = np.shape(img_1)
        
        imgs = []
        cv2.namedWindow(window_name)
        #视频来源，可以来自一段已存好的视频，也可以直接来自USB摄像头
        cap = cv2.VideoCapture(camera_idx)     
        classfier = cv2.CascadeClassifier(classifier_path)
        #识别出人脸后要画的边框的颜色，RGB格式
        color = (0, 255, 0)
        while cap.isOpened():
            ok, frame = cap.read() #读取一帧数据
            if not ok:            
                break                    
            #将当前帧转换成灰度图像
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                 
            #人脸检测，1.2和5分别为图片缩放比例和需要检测的有效点数
            faceRects = classfier.detectMultiScale(
                        gray,
                        scaleFactor=1.2,
                        minNeighbors=5,
                        minSize=(30, 30),
                        flags = 0)
            if len(faceRects) > 0:              #大于0则检测到人脸                                   
                for faceRect in faceRects:      #单独框出每一张人脸
                    x, y, w, h = faceRect        
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
                    cv2.ellipse(frame,(int(x+w/2),int(y+h/2-0.025*h)),
                                (int(74*(w+26)/200),int(90*(h+26)/200)),0,0,360,255,1)#加偏置是因为线宽原因
            #显示图像
            cv2.imshow(window_name, frame)        
            c = cv2.waitKey(10)
            #从视频中裁剪出人脸图片
            if c & 0xFF == ord('x'):
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                image = np.dot(image[...,:3], [0.299, 0.587, 0.114])
                img_2 = transform.resize(image,(100,100))
                for i in range(img_raw):
                    for j in range(img_col):
                        if img_1[i][j] == 0:
                            img_2[i,j]=0
                
                new_img = np.zeros((img_raw-11,img_col-27))
                new_img_raw,new_img_col = np.shape(new_img)
                for i in range(new_img_raw):
                    for j in range(new_img_col):
                        new_img[i][j] = img_2[i+3][j+13]
                pplt.imshow(new_img,cmap='gray')
                images = transform.resize(new_img,(44,36,1))
                imgs.append(images)
                break
    
        #释放摄像头并销毁所有窗口
        cap.release()
        cv2.destroyAllWindows() 
        return imgs,images
    
    imgs,img = catch_video("Get Video Stream",0)
#    
#a = np.array([8])
#new_y = sess.run([new_y],feed_dict = {x:imgs,y_:a,keep_prob:1.0})
#position = np.argmax(new_y)
#name = ['薛义权','宋锦涛','蓝高杰','陆权忠','覃绍彬','周俊成','黄凯','潘俊伟','陈东良']
#print('result ：',name[position]) 
#print(new_y)