# 8월 3일 실험내용 정리

1. Multi-module based Conditional VAE to predict the HVCM faults in SNS Accelerator 논문에서 제안한 모델 빛 학습방법에 대한 고찰.
1) 논문에서 제안한 모델을 이용해서 학습한 결과. 논문의 내용과 전혀 다른 양상들을 확인할 수 있었다.
2) 따라서, 논문에서 제안한 모델을 분석하기 위한 연구를 진행하였다.

2. 분석 방향
1) 논문에서 제안한 Loss 
본 논문에서 제안한 손실함수의 수식은 다음과 같다.

$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (x_i - \hat{x}_i)^2
$$
