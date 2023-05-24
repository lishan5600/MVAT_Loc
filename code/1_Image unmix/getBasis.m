%function W = getBasis( idx, datadir)
% W = GETBASIS( IDX)
% inputs:   IDX is the antibody id for a particular protein (default 1873)
% outputs:  W is the color basis matrix
% 
% Saves W to ./lib/2_imageProcessCode.
% 

clear 
clc

datadir = 'D:\IHCģ��\subcellular_location\���ݴ���\image';
%tissuedir = [datadir '/' num2str(idx) '/'];
tissuedir=datadir;
 d2 = dir(tissuedir); %��ȡ��ǰ�ļ��������е�ͼƬ����
% d2(1:2) = []; %ɾ��ǰ2��
count = 1;
[infor,name]=xlsread('image_name.xls');
for j=1:length(name)
    disp(j);
     infile= [tissuedir '/' name{j}];  
     I = imread( infile);
     counter = 1;
     eval('[W] = colorbasis( I);','counter = 0;');
        if counter
            Wbasis{count} = W;
            count = count + 1;
        end
end

W = zeros(size(Wbasis{1}));
for i=1:length(Wbasis)
    W = W + Wbasis{i};
end

save Wbasis_new.mat  W







