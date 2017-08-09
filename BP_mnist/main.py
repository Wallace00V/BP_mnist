# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 16:01:39 2017

@author: zwcong2
"""

 #本程序由UESTC的BigMoyan完成，并供所有人免费参考学习，但任何对本程序的使用必须包含这条声明
import math
import numpy as np
from input_data import input_data

################################################################################################
# 读入数据
################################################################################################
sample, label, test_s, test_l = input_data()
sample = np.array(sample,dtype='float') 
sample/= 256.0       # 特征向量归一化
test_s = np.array(test_s,dtype='float') 
test_s /= 256.0
##################################################################################################
# 神经网络配置
##################################################################################################
samp_num = len(sample)      # 样本总数
inp_num = len(sample[0])    # 输入层节点数
out_num = 10                # 输出节点数
hid_num = 15  # 隐层节点数(经验公式)
w1 = 0.2*np.random.random((inp_num, hid_num))- 0.1   # 初始化输入层权矩阵
w2 = 0.2*np.random.random((hid_num, out_num))- 0.1   # 初始化隐层权矩阵
hid_offset = np.zeros(hid_num)     # 隐层偏置向量
out_offset = np.zeros(out_num)     # 输出层偏置向量
inp_lrate = 0.2             # 输入层权值学习率
hid_lrate = 0.2             # 隐层学权值习率
###################################################################################################
 # sigmoid激活函数定义
###################################################################################################
def get_act(x):
    act_vec = []
    for i in x:
        act_vec.append(1/(1+math.exp(-i)))
    act_vec = np.array(act_vec)
    return act_vec
###################################################################################################
# 训练过程
###################################################################################################
for count in range(0, samp_num):
    t_label = np.zeros(out_num)
    t_label[label[count]] = 1
    
    #前向过程
    hid_value = np.dot(sample[count], w1) + hid_offset       # 隐层值
    hid_act = get_act(hid_value)                # 隐层激活值
    out_value = np.dot(hid_act, w2) + out_offset             # 输出层值
    out_act = get_act(out_value)                # 输出层激活值

    #后向过程
    e = t_label - out_act                          # 输出值与真值间的误差
    out_delta = e * out_act * (1-out_act)                                       # 输出层delta计算
    hid_delta = hid_act * (1-hid_act) * np.dot(w2, out_delta)                   # 隐层delta计算
    for i in range(0, out_num):
        w2[:,i] += hid_lrate * out_delta[i] * hid_act   # 更新隐层到输出层权向量
    for i in range(0, hid_num):
        w1[:,i] += inp_lrate * hid_delta[i] * sample[count]      # 更新输出层到隐层的权向量

    out_offset += hid_lrate * out_delta                             # 输出层偏置更新
    hid_offset += inp_lrate * hid_delta
print('Training Finished!')
###################################################################################################
#train_error
###################################################################################################
temp=0
for count in range(len(sample)):
    hid_value = np.dot(sample[count], w1) + hid_offset       # 隐层值
    hid_act = get_act(hid_value)                # 隐层激活值
    out_value = np.dot(hid_act, w2) + out_offset             # 输出层值
    out_act = get_act(out_value)                # 输出层激活值
    if np.argmax(out_act) == label[count]:
        temp+=1
print('Train_Set Error is: %.2f%%'%((1-float(temp)/len(sample))*100))
###################################################################################################
# test_error
###################################################################################################
temp=0
for count in range(len(test_s)):
    hid_value = np.dot(test_s[count], w1) + hid_offset       # 隐层值
    hid_act = get_act(hid_value)                # 隐层激活值
    out_value = np.dot(hid_act, w2) + out_offset             # 输出层值
    out_act = get_act(out_value)                # 输出层激活值
    if np.argmax(out_act) == test_l[count]:
        temp+=1
print('Test_Set Error is: %.2f%%'%((1-float(temp)/len(test_s))*100))
###################################################################################################
# 保存网络模型
###################################################################################################
Network = open("MyNetWork.txt", 'w')
Network.write(str(inp_num))
Network.write('\n')
Network.write(str(hid_num))
Network.write('\n')
Network.write(str(out_num))
Network.write('\n')
Network.write(str(inp_lrate)) 
Network.write('\n')      
Network.write(str(hid_lrate)) 
Network.write('\n')      
              
for i in w1:
    for j in i:
        Network.write(str(j))
        Network.write(' ')
    Network.write('\n')

for i in w2:
    for j in i:
        Network.write(str(j))
        Network.write(' ')
Network.write('\n')

for i in hid_offset:
    Network.write(str(i))
    Network.write(' ')
Network.write('\n')

for i in out_offset:
    Network.write(str(i))
    Network.write(' ')
Network.write('\n')
Network.close()