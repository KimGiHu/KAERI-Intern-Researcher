## 1. Mean and Standara from pretrained and proposed model
Pretrained Normal Loss - Mean:  3.4629488e-06  Std:  8.1137483e-07

Pretrained Fault Loss - Mean:  5.860476e-06  Std:  4.138254e-06

Proposed Normal Loss - Mean:  3.696645e-06  Std:  1.963688e-06

Proposed Fault Loss - Mean:  6.4978726e-06  Std:  5.4165002e-06


## 2. Confusion matrix of pretrained model & proposed mdoel
Confusion matrix | Predicted Negative  |  Predicted Positive
|------|---|---|
Actual Negative   |      TN            |         FP
Actual Positive   |      FN            |         TP

**Negative : Normal Signal**

**Positive : Abnormal Signal**

### 1. Pretrained model
ROC Curve에서 정상신호와 비정산신호를 구분하는 최적의 값 : 4.0755016925686505e-06

Precision: 0.7971

Recall: 0.4592

F1-score: 0.5827

Pretrained model Confusion matrix | Predicted Negative  |  Predicted Positive
|------|---|---|
Actual Negative   |     1199            |         199
Actual Positive   |     921            |         782

### 2. Proposed Model
ROC Curve에서 정상신호와 비정산신호를 구분하는 최적의 값 : 6.230558028619271e-06

Precision: 0.9554

Recall: 0.3400

F1-score: 0.5015

Proposed model Confusion matrix | Predicted Negative  |  Predicted Positive
|------|---|---|
Actual Negative   |     1371            |         27
Actual Positive   |     1124            |         579
