function [J] = linunmix(I, W)

W = double(W)/255;


% Prepare data for unmixing
S = size(I);
I = reshape( 255-I, [S(1)*S(2) S(3)]);

I = single(I);
I(I==0)=1e-9;

% Perform linear unmixing 
H = I*pinv(W)';
clear I

H = H - min(H(:));
H = H / max(H(:))*255;
H = uint8(round(H));

if 1==1
% Background correction
for i=1:size(H,2)
    [c b] = imhist(H(:,i));
    [a ind] = max(c);
    H(:,i)=H(:,i)-b(ind);
end
H=(255/max(H(:)))*H;
end

H = H * (255 / double(max(H(:))));
% Create output image
J = reshape( H, [S(1) S(2) size(W,2)]);
clear H S W

if size(J,3)<3
    J(:,:,3) = 0;
end
