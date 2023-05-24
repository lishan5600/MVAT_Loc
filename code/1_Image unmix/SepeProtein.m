clear all
clc
tic

%读取W矩阵
 load('Wbasis_new.mat')

%读取csx文件
%[data1,DATA]=xlsread('protein_normal_1.csv'); %DATA为字符串,1列为网址，2列为标签
%读取数据集文件夹下所有文件名

root_img='D:\IHC模型\subcellular_location\数据处理\image';
[infor,name]=xlsread('image_name.xls');

% imgname= dir(root_img); %读取当前文件夹下所有的图片数据
% imgname(1:2) = []; %删除前2行
%:length(imgname) 
for n=1:length(name)%数据集大小

    imgName_i=name{n};
    imgPath_i=strcat(root_img,'/',imgName_i);
    I = imread(imgPath_i);    %读取图片
    I = CleanBorders(I);                                                                                 
    I_unmixed = linunmix(I,W);    %输入图片和颜色基矩阵
    prot= I_unmixed(:,:,2);
    savePath='D:\IHC模型\subcellular_location\数据处理\分解图像\图像_分解';   %保存路径
    %saveMat=strcat(savePath,'/',imgName_i,'.mat');
    saveImg=strcat(savePath,'/',imgName_i);
         if ~exist(savePath)
             mkdir(savePath)
         end
         %save(saveMat,'prot')
         imwrite(prot, saveImg,'jpg');
         
     end
    toc