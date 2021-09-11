# -*- coding: utf-8 -*-
"""
Created on Tue May 29 10:50:05 2018

@author: Administrator
"""

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-1,2,50)
y1 = 2*x +1 
y2 = x**2

#method one
l1, =plt.plot(x,y1,'r-')
l2, =plt.plot(x,y2,'g--')
plt.legend(handles =[l1,l2,] , labels=['train_acc','test_acc'], loc = 'best')#或者可以自己调节图例的位置 loc=1 在右上角 =2左上角 =3 左下角 =4 右下角 一般 =‘best’ 自动调整
plt.xlabel('train_num')
plt.ylabel('acc')
plt.show()