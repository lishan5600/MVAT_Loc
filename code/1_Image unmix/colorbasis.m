function W = colorbasis( I, STOPCONN, A_THR, S_THR)
%% Methods for getting W are slightly more clearly defined in decomptissue.m.

rank = 2;  ITER = 5000;

tic;

if ~exist('STOPCONN','var')
    STOPCONN = 40;
end
if ~exist('A_THR','var')
    A_THR = 1000;
end
if ~exist('S_THR','var')
    S_THR = 1000;
end

I = 255 - I;
IMGSIZE = size(I);

% ....tissue size check!
if (IMGSIZE(1)<S_THR) || (IMGSIZE(2)<S_THR)
	error('Not enough useable tissue staining');
end


% ********** SEED COLORS ************
S = size(I);
V = reshape( I, S(1)*S(2),S(3));
[V ind VIDX] = unique(V,'rows');
VIDX = single(VIDX);

HSV = rgb2hsv( V);
hue = HSV(:,1);
[c b] = hist( hue(hue<0.3), [0:0.01:1]);
[A i] = max(c);
P = b(i);
hae = mean(V(P-.01<hue & hue<P+0.01,:),1)';

[c b] = hist( hue(hue>=0.3), [0:0.01:1]);
[A i] = max(c);
P = b(i);
dab = mean(V(P-.01<hue & hue<P+0.01,:),1)';

W = single( [hae dab] / 255);