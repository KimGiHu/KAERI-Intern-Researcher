식별 네트워크 변경사항.

식별자 모델구조 변경 :
1) 멀티 레이어 퍼셉트론 : 3개 -> 4개
기존모델      변경된 모델
(512,512)     (512,512)
(512,256)     (512,256)
(256,1)       (256,128)
              (128,1)

3) 학습률(Learning Rate) : 1e-5 -> 1e-3, 1e-4

- - - - - - - -
1. 기존 식별네트워크

![learning rate 1e-5 default model Normal vs  Fault ](https://github.com/user-attachments/assets/fdb1a0a4-d9f3-4bbe-b395-36a2d13c8d7a)


![default kdeplot Normal vs  Fault ](https://github.com/user-attachments/assets/8f915829-ef19-4d10-82c4-2c0490c75ef0)

- - - - - - - -
2. 변경한 식별네트워크 (학습률 : 1e-4)

![Total Normal vs  Fault_learning_Rate1e-4](https://github.com/user-attachments/assets/6f432714-b8bb-4927-a992-4824671f9735)

- - - - - - - -
3. 변경한 식별네트워크2 (학습률  1e-3)

![Total Normal vs  Fault _ test](https://github.com/user-attachments/assets/d38ed1e9-281d-4621-b146-39d8b9149f42)

![MLP_layer4_learning_rate_1e-3 Normal vs  Fault ](https://github.com/user-attachments/assets/411aed50-2b95-4d85-a41e-9071ad87ad09)
