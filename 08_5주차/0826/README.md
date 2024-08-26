# 실험내용

### Baseline Model - Case1

MSE Loss 함수를 사용할 때, 학습 시에 배치사이즈와 time step의 크기만큼 평균 내어서 학습한 경우

Baseline Normal Loss - Mean:  0.015485567  Std:  0.003634747

Baselin Fault Loss - Mean:  0.026580421  Std:  0.018456949

ROC Curve에서 정상신호와 비정산신호를 구분하는 최적의 값 : 0.017287295311689377

Precision: 0.7862

Recall: 0.5138

F1-score: 0.6214

Confusion Matrix:

[[1160  238]

 [ 828  875]]


![baseline Normal vs  Fault ROC Curve](https://github.com/user-attachments/assets/19061b81-f2c1-41b8-81ef-c94037a93f7c)

### Baseline Model - Case1

MSE Loss 함수를 사용할 때, reduction 기능을 'sum'으로 설정하고 학습한 경우

Baseline Normal Loss - Mean:  0.0031117005  Std:  0.0019744362

Baseline Fault Loss - Mean:  0.017567148  Std:  0.013441928

ROC Curve에서 정상신호와 비정산신호를 구분하는 최적의 값 : 0.004985465202480555

Precision: 0.8955

Recall: 0.9865

F1-score: 0.9388

Confusion Matrix:


[[1202  196]

 [  23 1680]]

![baseline_reduction_sum Normal vs  Fault ROC Curve](https://github.com/user-attachments/assets/5744835a-0871-44a4-9294-16235ca40bd7)

### Proposed Model

MSE Loss 함수를 사용할 때, reduction 기능을 'sum'으로 설정하고 학습한 경우 + 활성화 함수 gelu + 학습률 변경(1e-5 --> 1e-3) + 학습 스케쥴러 추가

Proposed Normal Loss - Mean:  0.0010912528  Std:  0.0005026587

Proposed Fault Loss - Mean:  0.020823535  Std:  0.027949354

ROC Curve에서 정상신호와 비정산신호를 구분하는 최적의 값 : 0.0033925846219062805

Precision: 0.9947

Recall: 0.9994

F1-score: 0.9971

Confusion Matrix:

[[1389    9]

 [   1 1702]]


![proposed Normal vs  Fault ROC Curve](https://github.com/user-attachments/assets/6b7f0803-8791-492e-9b3a-20753477dad3)


### 추가 실험건 : Loss function MSE를 reduction='mean'으로 실험하여봄.

Baseline Normal Loss - Mean:  0.0031117005  Std:  0.0019744362

Baseline Fault Loss - Mean:  0.017567148  Std:  0.013441928

Proposed Normal Loss - Mean:  0.01558474  Std:  0.003659449

Proposed Fault Loss - Mean:  0.026376178  Std:  0.018631684

ROC Curve에서 정상신호와 비정산신호를 구분하는 최적의 값 : 0.01839718595147133

Precision: 0.7998

Recall: 0.4551

F1-score: 0.5801

Confusion Matrix:

[[1204  194]

 [ 928  775]]

# 결론
본 실험에 사용한 데이터에는 MSE Loss를 계산할 때, batch size와 time step을 한 epoch마다 평균값을 내어 계산한 것(reduction을 'mean'으로 설정한 것)보다 전체 합(reduction을 'sum'으로 설정)을 한 다음 학습을 진행한 모델이 더 성능이 뛰어나다는걸 확인할 수 있었다. 이는 이상치를 정밀하게 탐지하기 위해 신호를 복원하여 비교하는 해당 모델의 특수성을 고려한다면 적합한 학습방식이라고 볼 수 있다. 또한 활성화함수를 'relu'에서 'gelu'로 변경하고, 학습률 조정 및 학습 스케쥴러 추가를 통해서 기존에 제안된 모델보다 향상된 F1-score 및 AUC를 얻을 수 있었다. 마지막으로, 기존 논문에서 제안된 방식보다 더 보편적인 데이터셋 비율을 학습에 사용하여서 실제 가속기 운영상황에서도 적용할 수 있을 것으로 고려된다.
