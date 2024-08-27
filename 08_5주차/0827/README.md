실험내용 총 정리를 위해서 종합적인 모델의 성능비교가 필요함.

# 1. 분류성능비교 : F1-score, AUC

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
  
- AUC : 1.0

  ![Baseline Model Normal vs  Fault ROC Curve](https://github.com/user-attachments/assets/290f9fd7-1822-4bab-9183-3447eab6c9d9)

## Proposed Model

**Reduction : Mean**

Proposed Model Precision: 0.7998
Proposed Model Recall: 0.4551
Proposed Model F1-score: 0.5801

- Mean and Loss
  Table | Mean | Standard Deviation
  |:-:|:-:|:-:|
  Normal Signal Loss |   0.01558 |   0.00366
  Fault Signal Loss |  0.02638  |  0.01863

- ROC Curve에서 정상신호와 비정산신호를 구분하는 최적의 값 : 0.01839718595147133
 
  by "Youden's J 통계량은 (TPR - FPR)의 최대값을 기준으로 최적의 threshold를 선택한다"
  
- Precision, Recall and F1-Score
- AUC : 1.0
2. 모델의 크기 및 최종 손실함수 값 비교
