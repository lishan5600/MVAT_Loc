# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 16:08:18 2023

@author: lisha
"""

from __future__ import print_function, division
#import torchsummary
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import pandas as pd
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import torch.optim as optim
from sklearn.metrics import accuracy_score,multilabel_confusion_matrix
from torch.optim import lr_scheduler
import npmetrics
#import mulroc_pr
import datainfor
import multi_model512_3 as multi_model_128
from src.loss_functions.losses_change import AsymmetricLoss,FocalLoss,AsymmetricLossOptimized,Asy1#引入损失函数
#导入数据并创建数据集
torch.manual_seed(1) #添加随机树种子
alexnet128_train= pd.read_csv("new_alexnet512_train_all.csv")
alexnet128_train=np.array(alexnet128_train) 
alexnet128_test= pd.read_csv("new_alexnet512_test_all.csv")
alexnet128_test=np.array(alexnet128_test) 

densent128_train= pd.read_csv("new_densnet512_train_all.csv")
densent128_train=np.array(densent128_train) 
densent128_test= pd.read_csv("new_densnet512_test_all.csv")
densent128_test=np.array(densent128_test) 

vgg128_train= pd.read_csv("new_vgg512_train_all.csv")
vgg128_train=np.array(vgg128_train) 
vgg128_test= pd.read_csv("new_vgg512_test_all.csv")
vgg128_test=np.array(vgg128_test) 

resnet128_train= pd.read_csv("new_resnet512_train_all.csv")
resnet128_train=np.array(resnet128_train) 
resnet128_test= pd.read_csv("new_resnet512_test_all.csv")
resnet128_test=np.array(resnet128_test)


lbp_train= pd.read_csv("new_lbp_train.CSV")
#lbp_train=pd.DataFrame(lbp_train,dtype=np.float)；
lbp_test= pd.read_csv("new_lbp_test.CSV")
lbp_train=np.array(lbp_train)
lbp_test=np.array(lbp_test)

m=resnet128_train.shape[1]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_class=5
# Transformer Parameters
d_model = 512# Embedding Size,512-1024,128-512
q_model=lbp_test.shape[1]
d_ff = 1024# FeedForward dimension
n_layers = 4  # number of Encoder of Decoder Layer
n_heads = 4  # number of heads in Multi-Head Attention
d_k = d_v =int(d_model/n_heads) # dimension of K(=Q), V
batch_size=50
#一些超参数的设计
a=2   #控制不对称的顺势函数，gamma_neg
b=2  #控制不对称的顺势函数，gamma_pos

lamda=0.01 #相关矩阵A的阈值 0:0.05:0.3
c=0.05 #转移概率 0，0.01：0.1
aa=0.2 #不平衡的控制，从0.05：0.05:0.95
# 定义自己数据集的数据读入类
#4个融合
#resnet128_train,vgg128_train,densent128_train,alexnet128_train,lbp_train
# num_fea=4
# train_Data = datainfor.my_Data_Set4(num_fea,m,resnet128_train,vgg128_train,densent128_train,alexnet128_train,lbp_train)
# test_Data = datainfor.my_Data_Set4(num_fea,m,resnet128_test,vgg128_test,densent128_test,alexnet128_test,lbp_test)
#还需要去修改multi_model_128的参数，num_fea=4。vgg128_test,densent128_test,alexnet128_test
#order=[resnet,densent,vgg,alexnet]

order1=1
order2=1
order3=0
order4=1
num_fea=3
train_Data = datainfor.my_Data_Set3(num_fea,m,resnet128_test,densent128_test,alexnet128_test,lbp_train)
test_Data = datainfor.my_Data_Set3(num_fea,m,resnet128_test,densent128_test,alexnet128_test,lbp_test)


#train_Data = my_Data_Set(vgg128_train,alexnet128_train,lbp_train)
# 读取数据集大小
dataset_sizes = {'train': train_Data.__len__(), 'test': test_Data.__len__()}
train_DataLoader = DataLoader(train_Data, batch_size, shuffle=True)
test_DataLoader = DataLoader(test_Data, batch_size, shuffle=True)
dataloaders = {'train':train_DataLoader, 'test':test_DataLoader}


def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()
    #active_fun = nn.Sigmoid()
    #best_model_wts = copy.deepcopy(model.state_dict())
    best_model_wts=[]
    best_acc = 0.0
    train_acc=0.0
    num_epoch=0
    train_pre_best=[]
    train_output_best=[]
    train_label_best=[]
    test_pre_best=[]
    test_output_best=[]
    test_label_best=[]
    for epoch in range(num_epochs):
        train_pre=[]
        train_output=[]
        train_label=[]
        test_pre=[]
        test_output=[]
        test_label=[]
        #ytesttmp=np.ones((1,num_class))*0.5
        train_pre=np.ones((1,num_class))*0.5
        train_output=np.ones((1,num_class))*0.5
        train_label=np.ones((1,num_class))*0.5
        test_pre=np.ones((1,num_class))*0.5
        test_output=np.ones((1,num_class))*0.5
        test_label=np.ones((1,num_class))*0.5
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for deepinputs,trainputs,labels in dataloaders[phase]:
                deepinputs = deepinputs.to(device)
                trainputs = trainputs.to(device)
                labels = labels.to(device)
                #print('输入trainputs：',trainputs.size())
                #print('输入deepinputs：',deepinputs.size())
                #print('输入labels：',labels.size())
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs,enc_self_attns= model(trainputs,deepinputs)
                    #print('输出outputs：',outputs.size())
                    #print(outputs.size())
                    outputs = outputs.to(device)
                    loss = criterion(outputs, labels)
                    # batch_size1=inputs.size(0)
                    # outputs=outputs.view(batch_size1,num_class)
                    preds=multi_model_128.preclass(outputs,lamda)
                    #loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                # statistics
                running_loss += loss.item() * deepinputs.size(0)
                #running_corrects += torch.sum(preds == labels.data)
                running_corrects += multi_model_128.newacc(preds,labels.data)
                #print('running_corrects',running_corrects)
                if phase == 'train':
                    preds=preds.detach().numpy()
                    outputs=outputs.detach().numpy()
                    labels=labels.detach().numpy()
                    train_pre=np.vstack((train_pre,preds))
                    train_output=np.vstack((train_output,outputs))
                    train_label=np.vstack((train_label,labels))
                if phase == 'test':
                    preds=preds.detach().numpy()
                    outputs=outputs.detach().numpy()
                    labels=labels.detach().numpy()
                    test_pre=np.vstack((test_pre,preds))
                    test_output=np.vstack((test_output,outputs))
                    test_label=np.vstack((test_label,labels))
            if phase == 'train':
                scheduler.step()
            epoch_loss = running_loss / dataset_sizes[phase]
            #torch.sum(preds == labels.data)
            #epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase == 'train':
                epoch_train_acc = running_corrects / dataset_sizes[phase]
                epoch_acc=epoch_train_acc
            if phase == 'test':
                epoch_test_acc = running_corrects / dataset_sizes[phase]
                epoch_acc=epoch_test_acc
            print('running_corrects',running_corrects)
            print(f'dataset_sizes{phase}',dataset_sizes[phase])
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            # deep copy the model,val或者train
            
            if phase == 'test' and epoch_test_acc>= best_acc:
                best_acc = epoch_test_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_epoch=num_epoch
                train_acc = epoch_train_acc
                train_pre_best=train_pre
                train_output_best=train_output
                train_label_best=train_label
                test_pre_best=test_pre
                test_output_best=test_output
                test_label_best=test_label
            
                
        print()
        num_epoch=num_epoch+1
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'train Acc: {train_acc:4f}')
    print(f'Best test Acc: {best_acc:4f}')
    print(f'best_epoch: {best_epoch:4f}')
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model,train_pre_best,train_output_best,train_label_best,test_pre_best,test_output_best,test_label_best

model_ft = multi_model_128.Encoder()
model_ft=model_ft.to(device)

criterion = nn.MultiLabelSoftMarginLoss(reduction='mean')
#criterion = nn.BCELoss().to(device)
#criterion = AsymmetricLoss(gamma_neg=a, gamma_pos=b, clip=c, disable_torch_grad_focal_loss=True)  #损失函数
#criterion = AsymmetricLossOptimized(gamma_neg=a, gamma_pos=b, clip=c, disable_torch_grad_focal_loss=True)  #损失函数
#criterion = Asy1(alpha=aa, clip=c)
#criterion =FocalLoss(nn.BCEWithLogitsLoss(), gamma=2, alpha=0.25)

#result_name = "%s.txt" % '512_4fea_ASLoss(%a,%b)_adam_lamda0.01'
#result_name = "%s.txt" % 'focalloss,gamma=2, alpha=0.2,lr=0.001, momentum=0.4,Adamax'
#model_dir='./results'
#result_path = os.path.join(model_dir, result_name)
#criterion =FocalLoss()
#result_name = "%s.txt" % '512_4fea_ASLoss(1,3)_adam_lamda0.01'
# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.4)
#不好
#optimizer_ft=optim.ASGD(model_ft.parameters(), lr=0.01, lambd=0.001, alpha=0.75, t0=1000000.0, weight_decay=0)
#optimizer_ft =optim.Rprop(model_ft.parameters(), lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50))
#optimizer_ft =optim.Adagrad(model_ft.parameters(), lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0)
#还可以
#optimizer_ft =optim.Adadelta(model_ft.parameters(), lr=0.8, rho=0.1, eps=1e-06, weight_decay=0)
#optimizer_ft=optim.RMSprop(model_ft.parameters(), lr=0.1, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
#optimizer_ft=optim.Adam(model_ft.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
#optimizer_ft=optim.Adamax(model_ft.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

#优化器的设置
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
model_ft,train_pre,train_output,train_label,test_pre,test_output,test_label = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=40)

#npmetrics.write_metrics(test_label[1:,:],test_pre[1:,:], result_path)
import index
ex_subset_acc,OAA,ex_acc,ex_precision,ex_recall,ex_f1,lab_acc_macro,lab_precision_macro,lab_recall_macro,lab_f1_macro,lab_acc_micro,lab_precision_micro,lab_recall_micro,lab_f1_micro,score_f1_macro,score_f1_micro,h_loss,z_o_loss,mAP,mul_matri=index.write_metrics(test_label[1:,:],test_pre[1:,:])
#a_result_ASL=[lamda,aa,c,ex_subset_acc, OAA,ex_acc,ex_recall,ex_f1,lab_acc_macro,lab_precision_macro,lab_recall_macro,lab_f1_macro,lab_acc_micro,lab_precision_micro,lab_recall_micro,lab_f1_micro,score_f1_macro,score_f1_micro,h_loss,z_o_loss,mAP,mul_matri]
a_result_ASL=[order1,order2,order3,order4,lamda,aa,c,ex_subset_acc, OAA,ex_acc,ex_recall,ex_f1,lab_acc_macro,lab_precision_macro,lab_recall_macro,lab_f1_macro,lab_acc_micro,lab_precision_micro,lab_recall_micro,lab_f1_micro,score_f1_macro,score_f1_micro,h_loss,z_o_loss,mAP,mul_matri]

#分别计算每一类的混淆矩阵
# from sklearn.metrics import  confusion_matrix
# m1=confusion_matrix(test_label[1:,0],test_pre[1:,0])
# m2=confusion_matrix(test_label[1:,1],test_pre[1:,1])
# m3=confusion_matrix(test_label[1:,2],test_pre[1:,2])
# m4=confusion_matrix(test_label[1:,3],test_pre[1:,3])
# m5=confusion_matrix(test_label[1:,4],test_pre[1:,4])

#torch.save(model_ft,'./model_transformer.pt')
mulroc_pr.plot_pr_multi_label(num_class,test_label[1:,:], test_output[1:,:])
mulroc_pr.plot_roc_multi_label(num_class,test_label[1:,:], test_output[1:,:])

# train_output1=pd.DataFrame(train_output[1:,:])
# train_output1.to_csv('训练得分1.csv')

# train_label1=pd.DataFrame(train_label[1:,:])
# train_label1.to_csv('训练标签1.csv')

# test_output1=pd.DataFrame(test_output[1:,:])
# test_output1.to_csv('att3_test_output.csv')

# test_label1=pd.DataFrame(test_label[1:,:])
# test_label1.to_csv('att3_test_label.csv')

# test_pre1=pd.DataFrame(test_pre[1:,:])
# test_pre1.to_csv('att3 _test_pre.csv')







