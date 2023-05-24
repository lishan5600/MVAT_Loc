function centers_pnas = findPatches1(I,prot,radius,numRegions)

field_shape = 'square';
shift = floor(radius/3);

% CREATE THE GRID 
shiftx = 0; shifty = 0; gap = 10/2;
[img_x, img_y, img_z] = size(I);

dimx = round(img_x/(radius*2)); dimy = round(img_y/(radius*2)); dim = min(dimx, dimy);
grid_std = zeros(dim,dim,2);

iG2 = 1; iG1 = 1;
grid_std(iG1,:,1) = repmat(radius+5,[dim,1]);
grid_std(:,iG2,2) = repmat(radius+5,[dim,1]);

for iG1 = 2:dim
    iG2 = iG1;
    grid_std(iG1,:,1) = 2*gap + grid_std(iG1-1,iG2,1)+2*radius;
    grid_std(:,iG2,2) = 2*gap + grid_std(1,iG2-1,2)+2*radius;
end

y_ind = find(img_y-radius-gap > grid_std(:,1,1));
x_ind = find(img_x-radius-gap > grid_std(1,:,2));
grid = zeros(x_ind(end),y_ind(end),2);
grid(:,:,:) = grid_std(1:length(x_ind),1:length(y_ind),:);
%
centers = []; I_used = I; I_worked = I_used;
prot_orig = prot; %nuc_orig = nuc;

 rad_disk = zeros(2*radius+1)+1;
Region_coord = [];

for iR = 1:numRegions
        ic = 0; ic2 = 0;
        iC = grid(1,1,1); iC2 = grid(1,1,2); 
        iC_array = [1:1:size(grid,1)];
        iC2_array = [1:1:size(grid,2)];
        PROT = [];
        centers = [];
        while iC < size(I,1)-radius-shift
            ic = ic + 1; ic2 = 0; iC = iC + shift; iC2 = grid(1,1,2);
            while iC2 < size(I,2)-radius-shift 
                ic2 = ic2 + 1; iC2 = iC2 + shift;
                iarray1 = []; array2 = [];
                img1_win = []; img2_win = [];
                cen_x = iC; cen_y = iC2;
                %N = rad_disk.* double(nuc(cen_x-radius:cen_x+radius,cen_y-radius:cen_y+radius ));
                %NUC(ic,ic2) = sum(sum(N));
                P = rad_disk .* double(prot(cen_x-radius:cen_x+radius,cen_y-radius:cen_y+radius ));
                PROT(ic,ic2) = sum(sum(P));
                centers(ic,ic2,1) = iC;
                centers(ic,ic2,2) = iC2; 
                I_worked(cen_x-radius:cen_x+radius,cen_y-radius:cen_y+radius,3) = I_worked(cen_x-radius:cen_x+radius,cen_y-radius:cen_y+radius,3) .* 0;
            end
        end
        [v,index] = max(PROT);
        [v1,index1] = max(max(PROT));
        X = index(index1);
        Y = index1;
       
        cen_x = centers(X,Y,1);
        cen_y = centers(X,Y,2);

    I_used(cen_x-radius:cen_x+radius,cen_y-radius:cen_y+radius,:) = I_used(cen_x-radius:cen_x+radius,cen_y-radius:cen_y+radius,:) .* repmat(uint8(~rad_disk),[1,1,3]);

    %nuc_field = nuc_orig(cen_x-radius:cen_x+radius,cen_y-radius:cen_y+radius) .* uint8(rad_disk);
    prot_field = prot_orig(cen_x-radius:cen_x+radius,cen_y-radius:cen_y+radius) .* uint8(rad_disk);

   %nuc(cen_x-radius:cen_x+radius,cen_y-radius:cen_y+radius) = nuc(cen_x-radius:cen_x+radius,cen_y-radius:cen_y+radius) .* uint8(~rad_disk);
   prot(cen_x-radius:cen_x+radius,cen_y-radius:cen_y+radius) = prot(cen_x-radius:cen_x+radius,cen_y-radius:cen_y+radius) .* uint8(~rad_disk);

    %Region(iR) = PROT(index(index1),index1);

    I = double(I_used);

    Region_coord(iR,1) = cen_x;
    Region_coord(iR,2) = cen_y;
end
centers_pnas = Region_coord;
