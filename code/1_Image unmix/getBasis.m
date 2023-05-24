%function W = getBasis( idx, datadir)
% W = GETBASIS( IDX)
% inputs:   IDX is the antibody id for a particular protein (default 1873)
% outputs:  W is the color basis matrix
% 
% Saves W to ./lib/2_imageProcessCode.
% 

clear 
clc

datadir = 'D:\IHC模型\subcellular_location\数据处理\image';
%tissuedir = [datadir '/' num2str(idx) '/'];
tissuedir=datadir;
 d2 = dir(tissuedir); %读取当前文件夹下所有的图片数据
% d2(1:2) = []; %删除前2行
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







