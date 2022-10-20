(작성 중)

***

### Abstract   

DNN은 Image Restoration에서 많이 사용된다.  
&emsp; Loss Criteration(?)은 대개 l2를 이용  
&emsp; l2는 더 큰 에러를 penalize 한다.  
&emsp; 이를 피하기 위해 l1이 활용됨

저자는 네트워크를 복구하기 위한 새로운 Loss Function을 제안  
&emsp; Riemannian Manifold(리만 다양체)에서 Geodesic Distance를 측정  
&emsp; l1의 뛰어난 특성을 활용

l1과 l2는 픽셀 거리를 반영 / Riemannian Loss는 이미지의 구조적 거리를 반영

Riemannian Loss는  
&emsp; l1 loss의 robutness(?)를 예방   
&emsp; 이미지의 contrast 반영

Experimental Results  
&emsp; 객관적 품질과 지각 품질 모두에 따라 보다 정확한 재구성[

***

### Introduction

