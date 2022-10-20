(작성 중)


***


### Abstract</br></br>

DNN은 Image Restoration에서 많이 사용된다.  
&emsp; - Loss Criteration(?)은 대개 l2를 이용  
&emsp; - l2는 더 큰 에러를 penalize 한다.  
&emsp; - 이를 피하기 위해 l1이 활용됨

저자는 네트워크를 복구하기 위한 새로운 Loss Function을 제안  
&emsp; - Riemannian Manifold(리만 다양체)에서 Geodesic Distance(최단 거리)를 측정  
&emsp; - l1의 뛰어난 특성을 활용

l1과 l2는 픽셀 거리를 반영 / Riemannian Loss는 이미지의 구조적 거리를 반영

Riemannian Loss는  
&emsp; - l1 loss의 robutness(?)를 예방   
&emsp; - 이미지의 contrast 반영

Experimental Results  
&emsp; - 객관적 품질과 지각 품질 모두에 따라 보다 정확한 재구성


***


### Introduction</br></br>

l2 Loss (MSE)는 PSNR의 주요 측정값이다.  
&emsp; - 하지만, white gaussian noise에 한해 작동  
&emsp; - 현실에서는 효과가 없음  
&emsp; - outlier는(극단치)는 l2 loss의 가중치 할당에 큰 영향을 미침  
&emsp;&emsp; - 네트워크 성능을 낮춤

해당 논문은 이미지의 Geodesic Distance를 측정하는 Loss Function을 제안
&emsp; - l1 loss의 성공을 이용  
&emsp; - Geodesic Distance의 l1 norm을 측정

LOG (Log-Euclidean Metric)
&emsp; - Fillard가 정의
&emsp; - symmetric positive definite(?)의 새로운 행렬 구성
&emsp; - Riemnannian metrics에 의해 
