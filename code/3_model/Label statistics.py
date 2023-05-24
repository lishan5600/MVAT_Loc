# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 14:46:14 2022

@author: lisha
"""
import pickle
import numpy as np
result = pickle.load(open('D:/ML_GCN/two/mlgcn/data/data/coco/coco_adj.pkl', 'rb'))
_adj = result['adj']

_nums = result['nums']
'''
1. 提前统计过的一个共生标签矩阵M，M[i] [j]的值是代表作者自己提前统计的标签i与标签j
一起出现的次数,adj就是M矩阵,
2. nums[i]代表的是标签i在整个数据集中共出现的次数
'''

_nums = _nums[:, np.newaxis]
_adj = _adj / _nums
#print(_adj)
'''
共生次数除以所有标签一共出现的次数
但是这样的矩阵可能会出现相关问题（参见论文），
这里作者认为还需要对该矩阵进行二值化，t是我们自己设置的一个阈值，
即概率小于t都是0，大于t则为1，
在作者提供的源码中运用于coco数据集(我在最终训练的时候采用的数据集也是coco)为t为0.4，
所以做处理如下:
'''
t=0.4#这里对应论文中的二值化
_adj[_adj < t] = 0
_adj[_adj >= t] = 1
'''
重新加权方案
'''
_adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
_adj = _adj + np.identity(80, int)  #这里的80表示类别
'''
以上程序对应于util中的def gen_A(num_classes, t, adj_file)函数
到了这一步，我们的A矩阵可以说是已经完成处理了，
接下来我们再像一般图卷积神经网络处理矩阵A得出我们最终用来聚合特征的矩阵Lsym
'''
import torch
_adj=torch.FloatTensor(_adj)
D = torch.pow(_adj.sum(1).float(), -0.5)
D = torch.diag(D)
adj = torch.matmul(torch.matmul(_adj, D).t(), D)
print(adj)
print(adj.shape)
'''
可以看到这是一个归一化后的对角矩阵，
至此我们在图卷积神经网络中聚合节点特征的矩阵就完成了，
那么接下来我们就可以完成我们图卷积神经网络的搭建
'''


result1 = pickle.load(open('D:/ML_GCN/two/mlgcn/data/data/coco/coco_glove_word2vec.pkl', 'rb'))