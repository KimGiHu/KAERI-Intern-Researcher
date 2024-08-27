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
# 2. 모델의 구성 및 계산복잡도 비교

## Baseline Model Summary

| Layer (type:depth-idx)   | Output Shape        | Param #   |
|--------------------------|---------------------|-----------|
| CVAE_baseline            | [2, 14, 4500]       | --        |
| ├─Encoder_baseline: 1-1  | [2, 512]            | --        |
| │ └─Conv1d: 2-1          | [2, 128, 4501]      | 21,632    |
| │ └─BatchNorm1d: 2-2     | [2, 128, 4501]      | 256       |
| │ └─MaxPool1d: 2-3       | [2, 128, 2245]      | --        |
| │ └─Dropout: 2-4         | [2, 128, 2245]      | --        |
| │ └─Conv1d: 2-5          | [2, 128, 2246]      | 196,736   |
| │ └─BatchNorm1d: 2-6     | [2, 128, 2246]      | 256       |
| │ └─MaxPool1d: 2-7       | [2, 128, 1118]      | --        |
| │ └─Dropout: 2-8         | [2, 128, 1118]      | --        |
| │ └─Conv1d: 2-9          | [2, 128, 1119]      | 196,736   |
| │ └─BatchNorm1d: 2-10    | [2, 128, 1119]      | 256       |
| │ └─MaxPool1d: 2-11      | [2, 128, 554]       | --        |
| │ └─Dropout: 2-12        | [2, 128, 554]       | --        |
| │ └─Linear: 2-13         | [2, 512]            | 36,307,456|
| │ └─BatchNorm1d: 2-14    | [2, 527]            | 1,054     |
| │ └─Linear: 2-15         | [2, 512]            | 270,336   |
| │ └─Linear: 2-16         | [2, 512]            | 270,336   |
| ├─Decoder_baseline: 1-2  | [2, 14, 4500]       | --        |
| │ └─Linear: 2-17         | [2, 512]            | 270,336   |
| │ └─Linear: 2-18         | [2, 70912]          | 36,377,856|
| │ └─BatchNorm1d: 2-19    | [2, 70912]          | 141,824   |
| │ └─Upsample: 2-20       | [2, 128, 1108]      | --        |
| │ └─BatchNorm1d: 2-21    | [2, 128, 1108]      | 256       |
| │ └─ConvTranspose1d: 2-22| [2, 128, 1107]      | 196,736   |
| │ └─Dropout: 2-23        | [2, 128, 1107]      | --        |
| │ └─Upsample: 2-24       | [2, 128, 2214]      | --        |
| │ └─BatchNorm1d: 2-25    | [2, 128, 2214]      | 256       |
| │ └─ConvTranspose1d: 2-26| [2, 128, 2213]      | 196,736   |
| │ └─Dropout: 2-27        | [2, 128, 2213]      | --        |
| │ └─Upsample: 2-28       | [2, 128, 4491]      | --        |
| │ └─BatchNorm1d: 2-29    | [2, 128, 4491]      | 256       |
| │ └─ConvTranspose1d: 2-30| [2, 14, 4500]       | 21,518    |

**activation function** : ReLU

**Total params**: 74,470,828  
**Trainable params**: 74,470,828  
**Non-trainable params**: 0  
**Total mult-adds (G)**: 3.17  
**Input size (MB)**: 0.50  
**Forward/backward pass size (MB)**: 58.34  
**Params size (MB)**: 297.88  
**Estimated Total Size (MB)**: 356.73  


### Proposed Model Summary
# Proposed Model Summary

| Layer (type:depth-idx)   | Output Shape        | Param #     |
|--------------------------|---------------------|-------------|
| CVAE_proposed                 | [2, 14, 4500]       | --          |
| ├─Encoder_proposed: 1-1       | [2, 512]            | --          |
| │ └─Conv1d: 2-1          | [2, 128, 4501]      | 21,632      |
| │ └─BatchNorm1d: 2-2     | [2, 128, 4501]      | 256         |
| │ └─MaxPool1d: 2-3       | [2, 128, 2245]      | --          |
| │ └─Dropout: 2-4         | [2, 128, 2245]      | --          |
| │ └─Conv1d: 2-5          | [2, 128, 2246]      | 196,736     |
| │ └─BatchNorm1d: 2-6     | [2, 128, 2246]      | 256         |
| │ └─MaxPool1d: 2-7       | [2, 128, 1118]      | --          |
| │ └─Dropout: 2-8         | [2, 128, 1118]      | --          |
| │ └─Conv1d: 2-9          | [2, 128, 1119]      | 196,736     |
| │ └─BatchNorm1d: 2-10    | [2, 128, 1119]      | 256         |
| │ └─MaxPool1d: 2-11      | [2, 128, 554]       | --          |
| │ └─Dropout: 2-12        | [2, 128, 554]       | --          |
| │ └─Linear: 2-13         | [2, 512]            | 36,307,456  |
| │ └─BatchNorm1d: 2-14    | [2, 527]            | 1,054       |
| │ └─Linear: 2-15         | [2, 512]            | 270,336     |
| │ └─Linear: 2-16         | [2, 512]            | 270,336     |
| ├─Decoder_proposed: 1-2       | [2, 14, 4500]       | --          |
| │ └─Linear: 2-17         | [2, 512]            | 270,336     |
| │ └─Linear: 2-18         | [2, 70912]          | 36,377,856  |
| │ └─BatchNorm1d: 2-19    | [2, 70912]          | 141,824     |
| │ └─Upsample: 2-20       | [2, 128, 1108]      | --          |
| │ └─BatchNorm1d: 2-21    | [2, 128, 1108]      | 256         |
| │ └─ConvTranspose1d: 2-22| [2, 128, 1107]      | 196,736     |
| │ └─Dropout: 2-23        | [2, 128, 1107]      | --          |
| │ └─Upsample: 2-24       | [2, 128, 2214]      | --          |
| │ └─BatchNorm1d: 2-25    | [2, 128, 2214]      | 256         |
| │ └─ConvTranspose1d: 2-26| [2, 128, 2213]      | 196,736     |
| │ └─Dropout: 2-27        | [2, 128, 2213]      | --          |
| │ └─Upsample: 2-28       | [2, 128, 4491]      | --          |
| │ └─BatchNorm1d: 2-29    | [2, 128, 4491]      | 256         |
| │ └─ConvTranspose1d: 2-30| [2, 14, 4500]       | 21,518      |

**activation function : GELU**

**Total params**: 74,470,828  
**Trainable params**: 74,470,828  
**Non-trainable params**: 0  
**Total mult-adds (G)**: 3.17  
**Input size (MB)**: 0.50  
**Forward/backward pass size (MB)**: 58.34  
**Params size (MB)**: 297.88  
**Estimated Total Size (MB)**: 356.73  


**계산복잡도(FLOPs) 비교**
[INFO] Register count_convNd() for <class 'torch.nn.modules.conv.Conv1d'>.  
[INFO] Register count_normalization() for <class 'torch.nn.modules.batchnorm.BatchNorm1d'>.  
[INFO] Register zero_ops() for <class 'torch.nn.modules.pooling.MaxPool1d'>.  
[INFO] Register zero_ops() for <class 'torch.nn.modules.dropout.Dropout'>.  
[INFO] Register count_linear() for <class 'torch.nn.modules.linear.Linear'>.  
[INFO] Register count_upsample() for <class 'torch.nn.modules.upsampling.Upsample'>.  
[INFO] Register count_convNd() for <class 'torch.nn.modules.conv.ConvTranspose1d'>.  
[INFO] Register count_convNd() for <class 'torch.nn.modules.conv.Conv1d'>.  
[INFO] Register count_normalization() for <class 'torch.nn.modules.batchnorm.BatchNorm1d'>.  
[INFO] Register zero_ops() for <class 'torch.nn.modules.pooling.MaxPool1d'>.  
[INFO] Register zero_ops() for <class 'torch.nn.modules.dropout.Dropout'>.  
[INFO] Register count_linear() for <class 'torch.nn.modules.linear.Linear'>.  
[INFO] Register count_upsample() for <class 'torch.nn.modules.upsampling.Upsample'>.  
[INFO] Register count_convNd() for <class 'torch.nn.modules.conv.ConvTranspose1d'>.  
Baseline Model FLOPs: 3181237624.0, Parameters: 74470828.0  
Proposed Model FLOPs: 3181237624.0, Parameters: 74470828.0  
