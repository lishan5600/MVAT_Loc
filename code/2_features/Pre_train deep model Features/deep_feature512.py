

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import time
import os
import pandas as pd

# 定义数据的处理方式
data_transforms = {
    'train': transforms.Compose([
        # 将图像进行缩放，缩放为256*256
        transforms.Resize((256,256)),
        # 在256*256的图像上随机裁剪出227*227大小的图像用于训练
        #transforms.RandomResizedCrop(227),
        # 图像用于翻转
        #transforms.RandomHorizontalFlip(),
        # 转换成tensor向量
        transforms.ToTensor(),
        # 对图像进行归一化操作
        # [0.485, 0.456, 0.406]，RGB通道的均值与标准差
       #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
       transforms.Normalize([0.485], [0.229])
    ]),
    # 测试集需要中心裁剪，甚至不裁剪，直接缩放为224*224for，不需要翻转
    'val': transforms.Compose([
        transforms.Resize((256,256)),
        #transforms.CenterCrop(227),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transforms.Normalize([0.485], [0.229])
    ]),
}
 
 
# 定义数据读入
def Load_Image_Information(path):
    # 图像存储路径
    #image_Root_Dir = r'D:\IHC模型\subcellular_location\数据处理\image'
    image_Root_Dir = r'D:\IHC模型\程序subcellular_location\new数据处理\分解图像\图像_分解'
    # 获取图像的路径
    iamge_Dir = os.path.join(image_Root_Dir, path)
    # 以RGB格式打开图像
    # Pytorch DataLoader就是使用PIL所读取的图像格式
    # 建议就用这种方法读取图像，当读入灰度图像时convert('')
    #return Image.open(iamge_Dir).convert('RGB')
    return Image.open(iamge_Dir).convert('')
 

# 定义自己数据集的数据读入类
class my_Data_Set(nn.Module):
    def __init__(self, txt, transform=None, target_transform=None, loader=None):
        super(my_Data_Set, self).__init__()
        # 打开存储图像名与标签的txt文件
        fp = open(txt, 'r')
        images = []
        labels = []
        # 将图像名和图像标签对应存储起来
        for line in fp:
            line.strip('\n')
            line.rstrip()
            information = line.split()
            images.append(information[0]) #图片名字
            # 将标签信息由str类型转换为float类型
            labels.append([float(l) for l in information[1:len(information)]])
        self.images = images #图像名
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
 
    # 重写这个函数用来进行图像数据的读取
    def __getitem__(self, item):
        # 获取图像名和标签
        imageName = self.images[item]
        label = self.labels[item]
        # 读入图像信息
        image = self.loader(imageName)
        # 处理图像数据
        if self.transform is not None:
            image = self.transform(image)
        # 需要将标签转换为float类型，BCELoss只接受float类型
        label = torch.FloatTensor(label)
        return image, label, imageName
 
    # 重写这个函数，来看数据集中含有多少数据
    def __len__(self):
        return len(self.images)

# 生成Pytorch所需的DataLoader数据输入格式
train_Data = my_Data_Set('newtrain.txt', transform=data_transforms['train'], loader=Load_Image_Information)
val_Data = my_Data_Set('newtest.txt', transform=data_transforms['val'], loader=Load_Image_Information)
train_DataLoader = DataLoader(train_Data, batch_size=1, shuffle=False)
val_DataLoader = DataLoader(val_Data, batch_size=1,shuffle=False)
dataloaders = {'train':train_DataLoader, 'val':val_DataLoader}
# 读取数据集大小
dataset_sizes = {'train': train_Data.__len__(), 'val': val_Data.__len__()}

premodel = torch.load('./new_alexnet_0.05_modle40.pt',map_location=torch.device('cpu'))
print(premodel)
## 定义 hook 函数
embeddings = {}
def get_activation(name):
    def hook(model, input, output):
        # 如果你想feature的梯度能反向传播，那么去掉 detach（）
        embeddings[name] = output.detach()
    return hook
# 基础配置
from torch.utils.data import DataLoader
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 放置 hook
#premodel.avgpool.register_forward_hook(get_activation('avgpool'))
premodel.classifier.lin_1.register_forward_hook(get_activation('classifier.lin_1'))
premodel.eval()
train_feature = []
embeddings = {}

train_feature= []
train_name=[]
train_label=[]
for inputs, labels, imageName in train_DataLoader:
    inputs = inputs.to(device)
    premodel(inputs)
    train_feature.append(copy.deepcopy(embeddings['classifier.lin_1'].cpu().numpy()))
    #train_feature.append(copy.deepcopy(embeddings['avgpool'][0].cpu().numpy()))
    train_name.append(imageName)
    train_label.append(labels)
# 获取 test 的 embedding
test_feature= []
test_name=[]
test_label=[]
embeddings = {}
for inputs, labels,imageName in val_DataLoader:
    inputs = inputs.to(device)
    premodel(inputs)
    test_feature.append(copy.deepcopy(embeddings['classifier.lin_1'].cpu().numpy()))
    #test_feature.append(copy.deepcopy(embeddings['avgpool'][0].cpu().numpy()))
    #把图像名，标签，特征整合
    test_name.append(imageName)
    test_label.append(labels)


#如果是fc.lin_1整合信息
test_all=[]
for i in range(len(test_feature)):
    print(i)
    name_=list(test_name[i])
    label1=test_label[i].numpy()
    lab=[j for j in label1]
    fea=[j for j in test_feature[i]] 
    all_infor1=np.append(name_,fea)
    all_infor=np.append(all_infor1,lab)
    test_all.append(all_infor)
    
train_all=[]
all_infor1=[]
all_infor=[]
for i in range(len(train_feature)):
    #print(i)
    
    name_=list(train_name[i])
    label1=train_label[i].numpy()
    lab=[j for j in label1]
    fea=[j for j in train_feature[i]]
    
    all_infor1=np.append(name_,fea)
    all_infor=np.append(all_infor1,lab)
    train_all.append(all_infor) 
    
 
data_train=pd.DataFrame(train_all)
data_train.to_csv('new_alexnet512_train_all.csv')
data_test=pd.DataFrame(test_all)
data_test.to_csv('new_alexnet512_test_all.csv')

#如果是avgpool整合信息

    
# 如果在avgpool的hook，需要将 embedding 转换成一维向量
# for elem in train_feature:
#     emb_final_train.append([elem[i][0][0] for i in range(len(elem))])
# for elem in test_feature:
#     emb_final_test.append([elem[i][0][0] for i in range(len(elem))])




# train_infor= pd.read_csv("train_infor.csv")
# train_infor=np.array(train_infor) 
# test_infor= pd.read_csv("test_infor.csv")
# test_infor=np.array(test_infor) 
# train_identi=train_infor[:,1]
# test_identi=test_infor[:,1]
# train_dict = {}
# test_dict = {}

# for i in range(len(test_feature)):
#     test_dict[test_name[i]] = test_feature[i]
   
# for i in range(len(train_feature)):
#     train_dict[train_identi[i]] = train_feature[i]
#np.save('./train_features512_sigmoid40.npy',train_dict)
#np.save('./test_features512_sigmoid40.npy',test_dict)
