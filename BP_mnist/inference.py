# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 09:34:16 2017

@author: zwcong2
"""

import math
import numpy as np
import re
from PIL import Image
import os
#读取文件夹中图片数据
###################################################################################################
dir_pic = [os.path.join('Mnist_Picture/',f) for f in os.listdir('Mnist_Picture')]
f =[f for f in os.listdir('Mnist_Picture')]
test_s=np.zeros((len(dir_pic),28*28))
for i in range(len(dir_pic)):
    im = Image.open(dir_pic[i])
    im_data = np.array(im,dtype='float')
    test_s[int(re.split('.jpeg',f[i])[0]),:] = im_data.reshape(28*28)
test_s /= 256.0
###################################################################################################
#读取神经网络配置
##################################################################################################
Network = open("MyNetWork.txt", 'r')   #"MyNetWork"
arg=[]
while 1:
    line = Network.readline()
    if not line:
        break
    pass
    arg.append(line.strip())

inp_num = int(arg[0])                                                # 读取输入层节点数
hid_num = int(arg[1])                                                # 隐层节点数
out_num = int(arg[2])                                                # 输出节点数
inp_lrate = float(arg[3])                                             # 输入层权值学习率
hid_lrate = float(arg[4])                                             # 隐层学权值习率

w_temp=[]
for i in range(5,5+inp_num):
    s1 = re.split(' ',arg[i])
    for j in s1:
       w_temp.append(float(j))
w1 = np.array(w_temp).reshape(inp_num,int(len(w_temp)/inp_num))      # 读取输入层权矩阵

w_temp=[]
s1 = re.split(' ',arg[5+inp_num])
for j in s1:
    w_temp.append(float(j))
w2 = np.array(w_temp).reshape(int(len(w_temp)/out_num),out_num)     # 读取隐层权矩阵

off_temp=[]
s1 = re.split(' ',arg[5+inp_num+1])
for j in s1:
    off_temp.append(float(j))
hid_offset = np.array(off_temp)                                    # 读取隐层偏置向量

off_temp=[]
s1 = re.split(' ',arg[5+inp_num+2])
for j in s1:
    off_temp.append(float(j))
out_offset = np.array(off_temp)                                    # 输出层偏置向量
##################################################################################################

 # 必要函数定义
###################################################################################################
def get_act(x):
    act_vec = []
    for i in x:
        act_vec.append(1/(1+math.exp(-i)))
    act_vec = np.array(act_vec)
    return act_vec
###################################################################################################

#运行网络，输出预测结果
###################################################################################################
for cout in range(len(test_s)):
    hid_value = np.dot(test_s[cout], w1) + hid_offset   # 隐层值
    hid_act = get_act(hid_value)                        # 隐层激活值
    out_value = np.dot(hid_act, w2) + out_offset        # 输出层值
    out_act = get_act(out_value)                        # 输出层激活值
    print ('picture_index: %d'%cout,'predict:',np.argmax(out_act))
###################################################################################################

