clear;
clc;

img_nonN = imread('src/nightimage.png');
img_nonN = double(img_nonN);
img = img_nonN / 255; % 정규화

img_r = img(:,:,1);
img_g = img(:,:,2);
img_b = img(:,:,3);

[h,w,d] = size(img);

a = 0.95; % parameter µ
b = 0.98; % parameter ω

A = 1;

img_inv_r = 1 - img_r;
img_inv_g = 1 - img_r;
img_inv_b = 1 - img_r;

%%

% 8
img_inv_pad_r = padarray(img_inv_r, [1,1], "symmetric");
img_cDCM_r = ordfilt2(img_inv_pad_r, 5, ones(3,3));
img_cDCM_r(h+2,:) = [];
img_cDCM_r(1,:) = [];
img_cDCM_r(:,w+2) = [];
img_cDCM_r(:,1) = [];

img_inv_pad_g = padarray(img_inv_g, [1,1], "symmetric");
img_cDCM_g = ordfilt2(img_inv_pad_g, 5, ones(3,3));
img_cDCM_g(h+2,:) = [];
img_cDCM_g(1,:) = [];
img_cDCM_g(:,w+2) = [];
img_cDCM_g(:,1) = [];

img_inv_pad_b = padarray(img_inv_b, [1,1], "symmetric");
img_cDCM_b = ordfilt2(img_inv_pad_b, 5, ones(3,3));
img_cDCM_b(h+2,:) = [];
img_cDCM_b(1,:) = [];
img_cDCM_b(:,w+2) = [];
img_cDCM_b(:,1) = [];



% 10
img_DCM_pad_r = padarray(img_cDCM_r, [1,1], "symmetric");
img_med_r = ordfilt2(img_DCM_pad_r, 5, ones(3,3));
img_med_r(h+2,:) = [];
img_med_r(1,:) = [];
img_med_r(:,w+2) = [];
img_med_r(:,1) = [];

img_DCM_pad_g = padarray(img_cDCM_g, [1,1], "symmetric");
img_med_g = ordfilt2(img_DCM_pad_g, 5, ones(3,3));
img_med_g(h+2,:) = [];
img_med_g(1,:) = [];
img_med_g(:,w+2) = [];
img_med_g(:,1) = [];

img_DCM_pad_b = padarray(img_cDCM_b, [1,1], "symmetric");
img_med_b = ordfilt2(img_DCM_pad_b, 5, ones(3,3));
img_med_b(h+2,:) = [];
img_med_b(1,:) = [];
img_med_b(:,w+2) = [];
img_med_b(:,1) = [];



% 11
k_r = img_med_r - img_cDCM_r;
k_pad_r = padarray(k_r, [1,1], "symmetric");
img_det_r = ordfilt2(k_pad_r, 5, ones(3,3));
img_det_r(h+2,:) = [];
img_det_r(1,:) = [];
img_det_r(:,w+2) = [];
img_det_r(:,1) = [];
img_det_r = abs(img_det_r);

k_g = img_med_g - img_cDCM_g;
k_pad_g = padarray(k_g, [1,1], "symmetric");
img_det_g = ordfilt2(k_pad_g, 5, ones(3,3));
img_det_g(h+2,:) = [];
img_det_g(1,:) = [];
img_det_g(:,w+2) = [];
img_det_g(:,1) = [];
img_det_g = abs(img_det_g);

k_b = img_med_r - img_cDCM_b;
k_pad_b = padarray(k_b, [1,1], "symmetric");
img_det_b = ordfilt2(k_pad_b, 5, ones(3,3));
img_det_b(h+2,:) = [];
img_det_b(1,:) = [];
img_det_b(:,w+2) = [];
img_det_b(:,1) = [];
img_det_b = abs(img_det_b);



% 12
img_smo_r = img_med_r - img_det_r;
img_smo_g = img_med_g - img_det_g;
img_smo_b = img_med_b - img_det_b;



% 13
img_DCM_r = img_cDCM_r;
img_cDCM_k_r = img_cDCM_r * a;

for i = 1 : h
    for j = 1 : w
        if img_cDCM_k_r(i,j) < img_smo_r(i,j)
            img_DCM_r(i,j) = img_cDCM_k_r(i,j);
        else
            img_DCM_r(i,j) = img_smo_r(i, j);
        end
    end
end

img_DCM_g = img_cDCM_g;
img_cDCM_k_g = img_cDCM_g * a;

for i = 1 : h
    for j = 1 : w
        if img_cDCM_k_g(i,j) < img_smo_g(i,j)
            img_DCM_g(i,j) = img_cDCM_k_g(i,j);
        else
            img_DCM_g(i,j) = img_smo_g(i, j);
        end
    end
end

img_DCM_b = img_cDCM_b;
img_cDCM_k_b = img_cDCM_b * a;

for i = 1 : h
    for j = 1 : w
        if img_cDCM_k_b(i,j) < img_smo_b(i,j)
            img_DCM_b(i,j) = img_cDCM_k_b(i,j);
        else
            img_DCM_b(i,j) = img_smo_b(i, j);
        end
    end
end



% 14
trans_r = (1/255) - (b*img_DCM_r);
trans_g = (1/255) - (b*img_DCM_g);
trans_b = (1/255) - (b*img_DCM_b);



%15
result_inv_r = ((img_inv_r - A/255) / trans_r) + A/255;
result_inv_g = ((img_inv_g - A/255) / trans_g) + A/255;
result_inv_b = ((img_inv_b - A/255) / trans_b) + A/255;



%16
result_r = 1 - result_inv_r;
result_g = 1 - result_inv_g;
result_b = 1 - result_inv_b;

result(:,:,1) = result_r;
result(:,:,2) = result_g;
result(:,:,3) = result_b;

%%
figure(1);
subplot(2,2,1), imshow(result_r);
subplot(2,2,2), imshow(result_g);
subplot(2,2,3), imshow(result_b);
subplot(2,2,4), imshow(result);

figure(2);
subplot(2,2,1), imshow(img_DCM_r);
subplot(2,2,2), imshow(img_DCM_g);
subplot(2,2,3), imshow(img_DCM_b);








