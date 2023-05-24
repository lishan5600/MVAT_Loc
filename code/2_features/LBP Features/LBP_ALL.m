clear all
clc
[train_infor,trainname]=xlsread('train_data_0312.xls');
[test_infor,testname]=xlsread('test_data_0312.xls');
%批量导入数据
for i=2:length(trainname)
    name=trainname{i};
    I = imread(name); 
    spoints = [1, 0; 1, -1; 0, -1; -1, -1; -1, 0; -1, 1; 0, 1; 1, 1];
    lbpfeat_train(i-1,:) = lbp(I,spoints,0,'nh');
end
for i=2:length(testname)
    name1=testname{i};
    Il = imread(name1); 
    spoints = [1, 0; 1, -1; 0, -1; -1, -1; -1, 0; -1, 1; 0, 1; 1, 1];
    lbpfeat_test(i-1,:) = lbp(Il,spoints,0,'nh');
end
% xlswrite('lbp_train.xls',lbpfeat_train)
% xlswrite('lbp_test.xls',lbpfeat_test)


