# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 16:08:30 2023

@author: lisha
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd


class my_Data_Set4(nn.Module):
    def __init__(self, num_fea,m,infor1,infor2,infor3,infor4,lbp):
        super(my_Data_Set4, self).__init__()
        # 打开存储图像名与标签的txt文件
        self.images = infor1[:,1] #图像名
        self.feature1=infor1[:,2:m-5]
        self.feature2=infor2[:,2:m-5]
        self.feature3=infor3[:,2:m-5]
        self.feature4=infor4[:,2:m-5]
        self.lbp=lbp
        self.labels = infor1[:,m-5:]
        self.num_fea=num_fea
        print('4个融合')
    def __getitem__(self, item):
        # 获取图像名和标签
       imageName = self.images[item]
       label = self.labels[item]
       # 读入图像信息
       feature1=self.feature1[item]
       feature2=self.feature2[item]
       feature3=self.feature3[item]
       feature4=self.feature4[item]
       lbp=self.lbp[item]
       feature1= pd.to_numeric(feature1) #object格式转为数值格式
       feature2= pd.to_numeric(feature2)
       feature3= pd.to_numeric(feature3)
       feature4= pd.to_numeric(feature4)
       label = pd.to_numeric(label)
       lbp= pd.to_numeric(lbp)
       feature_=np.append(feature1,feature2)
       feature_=np.append(feature_,feature3)
       feature_=np.append(feature_,feature4)
       feature_=np.matrix(feature_)
       lbp=np.matrix(lbp)
       #格式的转换
       lbp=lbp.reshape(1,-1)
       feature=feature_.reshape(self.num_fea,-1)
       # 需要将标签转换为float类型，BCELoss只接受float类型
       label = torch.FloatTensor(label)
       feature= torch.FloatTensor(feature)
       lbp= torch.FloatTensor(lbp)
       # print(lbp.shape)
       # print(feature.shape)
       # print(label.shape)
       return feature,lbp,label
 
    # 重写这个函数，来看数据集中含有多少数据
    def __len__(self):
        return len(self.images)
#train_Data = my_Data_Set(resnet128_train,vgg128_train,densent128_train,alexnet128_train,lbp_train)
#test_Data = my_Data_Set(resnet128_test,vgg128_test,densent128_test,alexnet128_test,lbp_test)
#train_Data = my_Data_Set(vgg128_train,alexnet128_train,lbp_train)
#test_Data = my_Data_Set(vgg128_test,alexnet128_test,lbp_test)


#3个融合
class my_Data_Set3(nn.Module):
    
    def __init__(self, num_fea,m,infor1,infor2,infor3,lbp):
        super(my_Data_Set3, self).__init__()
        # 打开存储图像名与标签的txt文件
        self.images = infor1[:,1] #图像名
        self.feature1=infor1[:,2:m-5]
        self.feature2=infor2[:,2:m-5]
        self.feature3=infor3[:,2:m-5]
        self.lbp=lbp
        self.labels = infor1[:,m-5:]
        self.num_fea=num_fea
        print('3个融合')
    def __getitem__(self, item):
        # 获取图像名和标签
       imageName = self.images[item]
       label = self.labels[item]
       # 读入图像信息
       feature1=self.feature1[item]
       feature2=self.feature2[item]
       feature3=self.feature3[item]
       lbp=self.lbp[item]
       feature1= pd.to_numeric(feature1) #object格式转为数值格式
       feature2= pd.to_numeric(feature2)
       feature3= pd.to_numeric(feature3)
       label = pd.to_numeric(label)
       lbp= pd.to_numeric(lbp)
       feature_=np.append(feature1,feature2)
       feature_=np.append(feature_,feature3)
       feature_=np.matrix(feature_)
       lbp=np.matrix(lbp)
       #格式的转换
       lbp=lbp.reshape(1,-1)
       feature=feature_.reshape(self.num_fea,-1)
       # 需要将标签转换为float类型，BCELoss只接受float类型
       label = torch.FloatTensor(label)
       feature= torch.FloatTensor(feature)
       lbp= torch.FloatTensor(lbp)
       # print(lbp.shape)
       # print(feature.shape)
       # print(label.shape)
       return feature,lbp,label
 
    # 重写这个函数，来看数据集中含有多少数据
    def __len__(self):
        return len(self.images)

##2个融合
class my_Data_Set2(nn.Module):
    
    def __init__(self,num_fea,m, infor1,infor2,lbp):
        super(my_Data_Set2, self).__init__()
        # 打开存储图像名与标签的txt文件
        self.images = infor1[:,1] #图像名
        self.feature1=infor1[:,2:m-5]
        self.feature2=infor2[:,2:m-5]
        self.lbp=lbp
        self.labels = infor1[:,m-5:]
        self.num_fea=num_fea
        print('2个融合')
    def __getitem__(self, item):
        # 获取图像名和标签
       imageName = self.images[item]
       label = self.labels[item]
       # 读入图像信息
       feature1=self.feature1[item]
       feature2=self.feature2[item]
       lbp=self.lbp[item]
       feature1= pd.to_numeric(feature1) #object格式转为数值格式
       feature2= pd.to_numeric(feature2)
       label = pd.to_numeric(label)
       lbp= pd.to_numeric(lbp)
       feature_=np.append(feature1,feature2)
       feature_=np.matrix(feature_)
       lbp=np.matrix(lbp)
       #格式的转换
       lbp=lbp.reshape(1,-1)
       feature=feature_.reshape(self.num_fea,-1)
       # 需要将标签转换为float类型，BCELoss只接受float类型
       label = torch.FloatTensor(label)
       feature= torch.FloatTensor(feature)
       lbp= torch.FloatTensor(lbp)
       # print(lbp.shape)
       # print(feature.shape)
       # print(label.shape)
       return feature,lbp,label
 
    # 重写这个函数，来看数据集中含有多少数据
    def __len__(self):
        return len(self.images)