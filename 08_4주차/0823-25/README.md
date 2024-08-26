다양한 실험(학습 손실함수 조정, 학습률 및 스케쥴러 추가, 도메인 적응기법 네트워크 추가 등)을 진행한 끝에 다음과 같은 결과를 얻었다.

1. ROC Curve 그래프
![activation gelu pretrained Normal vs  Fault ROC Curve](https://github.com/user-attachments/assets/4d01cc2a-fbcc-40ca-871a-07d1a0933ed6)

2. 모델의 손실함수 평균(Mean) 및 표준편차(Standard Deviation)
1) Tunning model
Normal signal Loss - Mean:  0.00267589  Std:  0.0011680671
Fault signal Loss - Mean:  0.019644942  Std:  0.022774935

2) Domain-Adaptation mdoel
Normal signal Loss - Mean:  0.028704124  Std:  0.010606855
Fault signal Loss - Mean:  0.04257906  Std:  0.017675094

3. Precision, Recall and F1-score
1) Tunning Model

- Precision: 0.9571
- Recall: 0.9307
- F1-score: 0.9437

- Confusion Matrix:
  Tunning model | Predicted Negative  |  Predicted Positive
|:------:|:---:|:---:|
Actual Negative   |      1327            |         71
Actual Positive   |      118            |         1585

3) Domain-Adaptation(DA) model

- Precision: 0.8248
- Recall: 0.6442
- F1-score: 0.7234

- Confusion Matrix:
  DA model | Predicted Negative  |  Predicted Positive
|:------:|:---:|:---:|
Actual Negative   |      1327            |         71
Actual Positive   |      118            |         1585

[[1165  233]
 [ 606 1097]]



- - - - - - - -
실험한 내용들을 다음과 같은 분류항목들을 정리하고자 한다.

실험내용

1. 모델의 성능향상(F1-score 및 AUC  향상)

2. 모델의 경량화

2-1) 학습 파라미터 수
2-2) 학습 및 추론시간 비교
2-3) 모델 구조 비교
2-4) 모델 연산량 비교 - 계산복잡도(FLOPs)  
