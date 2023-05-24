

function [Inew] = CleanBorders(I) 

Itemp = sum(I,3);

Inew = I;
index = sum((Itemp==0),2)>30;
Inew(index,:,:) = [];

Itemp = sum(Inew,3);
index = sum((Itemp==0),1)>30;
Inew(:,index,:) = [];










