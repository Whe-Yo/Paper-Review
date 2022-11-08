clear;
clc;

img_nonN = imread('src/nightimage.png');
img_nonN = double(img_nonN);
img = img_nonN / 255; % 정규화

[h,w,d] = size(img);

a = 0.95; % parameter µ
b = 0.98; % parameter ω

A = 1;

img_inv = 1 - img;

%%

% 8
img_inv_pad = padarray(img_inv, [1,1], "symmetric");
img_cDCM(:,:,1) = ordfilt2(img_inv_pad(:,:,1), 5, ones(3,3));
img_cDCM(:,:,2) = ordfilt2(img_inv_pad(:,:,2), 5, ones(3,3));
img_cDCM(:,:,3) = ordfilt2(img_inv_pad(:,:,3), 5, ones(3,3));
img_cDCM(h+2,:,:) = [];
img_cDCM(1,:,:) = [];
img_cDCM(:,w+2,:) = [];
img_cDCM(:,1,:) = [];



% 10
img_DCM_pad = padarray(img_cDCM, [1,1], "symmetric");
img_med(:,:,1) = ordfilt2(img_DCM_pad(:,:,1), 5, ones(3,3));
img_med(:,:,2) = ordfilt2(img_DCM_pad(:,:,2), 5, ones(3,3));
img_med(:,:,3) = ordfilt2(img_DCM_pad(:,:,3), 5, ones(3,3));
img_med(h+2,:,:) = [];
img_med(1,:,:) = [];
img_med(:,w+2,:) = [];
img_med(:,1,:) = [];



% 11
k_r = img_med - img_cDCM;
k_pad_r = padarray(k_r, [1,1], "symmetric");
img_det(:,:,1) = ordfilt2(k_pad_r(:,:,1), 5, ones(3,3));
img_det(:,:,2) = ordfilt2(k_pad_r(:,:,2), 5, ones(3,3));
img_det(:,:,3) = ordfilt2(k_pad_r(:,:,3), 5, ones(3,3));
img_det(h+2,:,:) = [];
img_det(1,:,:) = [];
img_det(:,w+2,:) = [];
img_det(:,1,:) = [];
img_det = abs(img_det);



% 12
img_smo = img_med - img_det;



% 13
img_DCM = img_cDCM;
img_cDCM_k = img_cDCM * a;

for l = 1 : d
    for i = 1 : h
        for j = 1 : w
            if img_cDCM_k(i,j,d) < img_smo(i,j,d)
                img_DCM(i,j,d) = img_cDCM_k(i,j,d);
            else
                img_DCM(i,j,d) = img_smo(i,j,d);
            end
        end
    end
end



% 14
trans = 1 - (b*img_DCM);



%15
result_inv = img_inv;

for l = 1 : d
    for i = 1 : h
        for j = 1 : w
            result_inv(i,j,l) = ((img_inv(i,j,l) - A) / trans(i,j,l)) + A;
        end
    end
end



%16
result = 1 - result_inv;



%%
figure(1);
subplot(2,2,1), imshow(img);
subplot(2,2,2), imshow(result);






