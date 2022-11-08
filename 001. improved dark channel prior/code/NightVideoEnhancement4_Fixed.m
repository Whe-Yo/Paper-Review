clear;
clc;
close all;


% img = im2double('src/28.bmp');
% img = im2double('src/36.bmp');
% img = im2double('src/38.bmp');
% img = im2double('src/39.bmp');
% img = im2double('src/nightimage.png');
img = im2double(imread('src/38.bmp'));

result = NVE(img);
% result2 = NVE(img);
% result3 = NVE(img);
% result4 = NVE(img);
% result5 = NVE(img);

figure(1); imshow(result), title('result6');

figure(2); imshow(img), title('img6');


function result = NVE(img_nonN)

% img_nonN = double(img_nonN);
img = img_nonN; % 정규화

[h,w,d] = size(img);

a = 0.95; % parameter µ
b = 0.8; % parameter ω

A = 1;

img_inv = 1 - img;

dcm = min(img_inv(:,:,3), min(img_inv(:,:,1), img_inv(:,:,2)));

% 8



% img_inv_pad = padarray(img_inv, [1,1], "symmetric");
img_cDCM = ordfilt2(dcm, 1, ones(3,3), 'symmetric');
% img_cDCM(:,:,2) = ordfilt2(img_inv_pad(:,:,2), 1, ones(3,3));
% img_cDCM(:,:,3) = ordfilt2(img_inv_pad(:,:,3), 1, ones(3,3));
% img_cDCM(h+2,:,:) = [];
% img_cDCM(1,:,:) = [];
% img_cDCM(:,w+2,:) = [];
% img_cDCM(:,1,:) = [];



% 10
% img_DCM_pad = padarray(img_cDCM, [1,1], "symmetric");
img_med = ordfilt2(img_cDCM(:,:,1), 5, ones(3,3),'symmetric');
% img_med(:,:,2) = ordfilt2(img_DCM_pad(:,:,2), 5, ones(3,3));
% img_med(:,:,3) = ordfilt2(img_DCM_pad(:,:,3), 5, ones(3,3));
% img_med(h+2,:,:) = [];
% img_med(1,:,:) = [];
% img_med(:,w+2,:) = [];
% img_med(:,1,:) = [];



% 11
k_r = img_med - img_cDCM;
% k_pad_r = padarray(k_r, [1,1], "symmetric");
img_det = ordfilt2(k_r(:,:,1), 5, ones(3,3), 'symmetric');
% img_det(:,:,2) = ordfilt2(k_pad_r(:,:,2), 5, ones(3,3));
% img_det(:,:,3) = ordfilt2(k_pad_r(:,:,3), 5, ones(3,3));
% img_det(h+2,:,:) = [];
% img_det(1,:,:) = [];
% img_det(:,w+2,:) = [];
% img_det(:,1,:) = [];
img_det = abs(img_det);



% 12
img_smo = img_med - img_det;



% 13
img_DCM_k = img_cDCM;
img_cDCM_k = img_cDCM * a;

img_DCM = min(img_cDCM_k, img_smo);

% for i = 1 : h
%    for j = 1 : w
%        if img_cDCM_k(i,j,:) < img_smo(i,j,:)
%                 img_DCM_k(i,j,:) = img_cDCM_k(i,j,:);
%        else
%                 img_DCM_k(i,j,:) = img_smo(i,j,:);
%        end
%    end
% end


% img_DCM_k_pad = padarray(img_DCM_k, [1,1], "symmetric");
% img_DCM = ordfilt2(img_DCM_k_pad(:,:,1), 9, ones(3,3));
% % img_DCM(:,:,2) = ordfilt2(img_DCM_k_pad(:,:,2), 9, ones(3,3));
% % img_DCM(:,:,3) = ordfilt2(img_DCM_k_pad(:,:,3), 9, ones(3,3));
% img_DCM(h+2,:,:) = [];
% img_DCM(1,:,:) = [];
% img_DCM(:,w+2,:) = [];
% img_DCM(:,1,:) = [];



% 14
trans = 1 - (b*img_DCM);

result_inv = ((img_inv - A) ./ trans) + A;

%15
% result_inv = img_inv;

% for l = 1 : d
%     for i = 1 : h
%         for j = 1 : w
%             result_inv(i,j,l) = ((img_inv(i,j,l) - A) / trans(i,j,l)) + A;
%         end
%     end
% end



%16
result = 1 - result_inv;



end