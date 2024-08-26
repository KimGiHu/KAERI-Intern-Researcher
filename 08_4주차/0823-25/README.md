다양한 실험(학습 손실함수 조정, 학습률 및 스케쥴러 추가, 도메인 적응기법 네트워크 추가 등)을 진행한 끝에 다음과 같은 결과를 얻었다.

![activation gelu pretrained Normal vs  Fault ROC Curve](https://github.com/user-attachments/assets/4d01cc2a-fbcc-40ca-871a-07d1a0933ed6)


- - - - - - - -
실험한 내용들을 다음과 같은 분류항목들을 정리하고자 한다.

실험내용

1. 모델의 성능향상(F1-score 및 AUC  향상)

2. 모델의 경량화

2-1) 학습 파라미터 수
2-2) 학습 및 추론시간 비교
2-3) 모델 구조 비교
2-4) 모델 연산량 비교 - 계산복잡도(FLOPs)  
