learning rate 스케쥴러를 적용하여서 학습의 성능을 높이는 방법을 찾아봄.

일단은 다음과 같이 적용함.
>scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_source, mode='min', factor=0.3, patience=3)

pretrain 모델만 학습하고, adpatation 기법을 적용한 이전 모델을 사용함.



