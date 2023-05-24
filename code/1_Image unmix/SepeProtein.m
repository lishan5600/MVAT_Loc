clear all
clc
tic

%��ȡW����
 load('Wbasis_new.mat')

%��ȡcsx�ļ�
%[data1,DATA]=xlsread('protein_normal_1.csv'); %DATAΪ�ַ���,1��Ϊ��ַ��2��Ϊ��ǩ
%��ȡ���ݼ��ļ����������ļ���

root_img='D:\IHCģ��\subcellular_location\���ݴ���\image';
[infor,name]=xlsread('image_name.xls');

% imgname= dir(root_img); %��ȡ��ǰ�ļ��������е�ͼƬ����
% imgname(1:2) = []; %ɾ��ǰ2��
%:length(imgname) 
for n=1:length(name)%���ݼ���С

    imgName_i=name{n};
    imgPath_i=strcat(root_img,'/',imgName_i);
    I = imread(imgPath_i);    %��ȡͼƬ
    I = CleanBorders(I);                                                                                 
    I_unmixed = linunmix(I,W);    %����ͼƬ����ɫ������
    prot= I_unmixed(:,:,2);
    savePath='D:\IHCģ��\subcellular_location\���ݴ���\�ֽ�ͼ��\ͼ��_�ֽ�';   %����·��
    %saveMat=strcat(savePath,'/',imgName_i,'.mat');
    saveImg=strcat(savePath,'/',imgName_i);
         if ~exist(savePath)
             mkdir(savePath)
         end
         %save(saveMat,'prot')
         imwrite(prot, saveImg,'jpg');
         
     end
    toc