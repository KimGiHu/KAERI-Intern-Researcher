실험내용 총 정리를 위해서 종합적인 모델의 성능비교가 필요함.

# 1. 이상치 탐지 성능비교 : F1-score, AUC

정상신호와 비정상신호간의 Youden's J 통계량을 임계값(Threshold Value)로 설정하고, 이보다 크면 정상 작으면 비정상으로 분류한다.

## Baseline Model 

**Reduction : Mean**

- Mean and Loss
  
  Table | Mean | Standard Deviation
  |:-:|:-:|:-:|
  Normal Signal Loss |   0.01549 |   0.00363
  Fault Signal Loss |  0.02658  |  0.01846

- ROC Curve에서 정상신호와 비정산신호를 구분하는 최적의 값 : 0.017287295311689377
 
  by "Youden's J 통계량은 (TPR - FPR)의 최대값을 기준으로 최적의 threshold를 선택한다"
  
- Precision, Recall and F1-Score

  Baseline Model Precision : 875/(875+238) = 0.7862

  Baseline Model Recall : 875/(875+828) = 0.5138

  Baseline Model F1-score : 0.6214
  
  Table | Predicted Normal | Predicted Fault
  |:-:|:-:|:-:|
  Actual Normal |   1160 |   238
  Actual Fault |  828  |  875
  
- AUC : 0.73

  ![Baseline Model Normal vs  Fault ROC Curve](https://github.com/user-attachments/assets/290f9fd7-1822-4bab-9183-3447eab6c9d9)

**Reduction : Sum**

- Mean and Loss
  
  Table | Mean | Standard Deviation
  |:-:|:-:|:-:|
  Normal Signal Loss |   0.00311 |   0.00197
  Fault Signal Loss |  0.01757  |  0.01344

- ROC Curve에서 정상신호와 비정산신호를 구분하는 최적의 값 : 0.004985465202480555
 
  by "Youden's J 통계량은 (TPR - FPR)의 최대값을 기준으로 최적의 threshold를 선택한다"
  
- Precision, Recall and F1-Score

  Baseline Model Precision : 1680/(1680+238) = 0.8955  

  Baseline Model Recall : 1680/(1680+23) = 0.9865

  Baseline Model F1-score : 0.9388
  
  Table | Predicted Normal | Predicted Fault
  |:-:|:-:|:-:|
  Actual Normal |   1202 |   196
  Actual Fault |  23  |  1680
  
- AUC : 0.97

  ![Baseline Model Reduction-sum Normal vs  Fault ROC Curve](https://github.com/user-attachments/assets/03611210-583e-4a40-a42e-d4f097db65d0)

## Proposed Model

**Reduction : Mean**

- Mean and Loss
  Table | Mean | Standard Deviation
  |:-:|:-:|:-:|
  Normal Signal Loss |   0.01558 |   0.00366
  Fault Signal Loss |  0.02638  |  0.01863

- ROC Curve에서 정상신호와 비정산신호를 구분하는 최적의 값 : 0.01839718595147133
 
  by "Youden's J 통계량은 (TPR - FPR)의 최대값을 기준으로 최적의 threshold를 선택한다"
  
- Precision, Recall and F1-Score

  Proposed Model Precision: 0.7998

  Proposed Model Recall: 0.4551

  Proposed Model F1-score: 0.5801
  
  Table | Predicted Normal | Predicted Fault
  |:-:|:-:|:-:|
  Actual Normal |   1204 |   194
  Actual Fault |  928  |  775
  
- AUC : 0.70
  
    ![Proposed Model Normal vs  Fault ROC Curve](https://github.com/user-attachments/assets/269a972a-b465-4896-971a-12e47f1d5b39)
  

**Reduction : Sum**

- Mean and Loss
  
  Table | Mean | Standard Deviation
  |:-:|:-:|:-:|
  Normal Signal Loss |   0.00109 |   0.00050
  Fault Signal Loss |  0.02082  |  0.02795
  
- ROC Curve에서 정상신호와 비정산신호를 구분하는 최적의 값 : 0.0033925846219062805
 
  by "Youden's J 통계량은 (TPR - FPR)의 최대값을 기준으로 최적의 threshold를 선택한다"
  
- Precision, Recall and F1-Score

  Baseline Model Precision : 1702/(1702+9) = 0.9947  

  Baseline Model Recall : 1702/(1+1702) = 0.9994

  Baseline Model F1-score : 0.9971
  
  Table | Predicted Normal | Predicted Fault
  |:-:|:-:|:-:|
  Actual Normal |   1389 |   9
  Actual Fault |  1  |  1702

- AUC : 1.00
  
  ![Proposed Model Reduction-sum Normal vs  Fault ROC Curve](https://github.com/user-attachments/assets/de7d3901-5c9e-42ac-aee6-f9ec94acf702)

- - - - - -
# 2. 모델의 크기 및 최종 손실함수 값 비교
