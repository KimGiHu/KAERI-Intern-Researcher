실험내용 총 정리를 위해서 종합적인 모델의 성능비교가 필요함.

# 1. 분류성능비교 : F1-score, AUC

## 1. Baseline Model 

**Reduction : Mean**

Normal Signal Loss - Mean:  0.01549  Std:  0.00363

Fault Signal Loss - Mean:  0.02658  Std:  0.01846

ROC Curve에서 정상신호와 비정산신호를 구분하는 최적의 값 : 0.017287295311689377 by "Youden's J 통계량은 (TPR - FPR)의 최대값을 기준으로 최적의 threshold를 선택한다"

[ Proposed Model - Reduction : 'Mean']


2. 모델의 크기 및 최종 손실함수 값 비교
