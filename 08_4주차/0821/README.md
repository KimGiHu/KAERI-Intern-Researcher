adaptation 모델에도 학습률 스케쥴러를 다음과 같이 이용함.

### 학습률 스케쥴러 추가
> scheduler_g = optim.lr_scheduler.ReduceLROnPlateau(optimizer_target, mode='min', factor=0.3, patience=3)
> scheduler_d_mu = optim.lr_scheduler.ReduceLROnPlateau(optimizer_discriminator_logvar, mode='min', factor=0.3, patience=3)
> scheduler_d_logvar = optim.lr_scheduler.ReduceLROnPlateau(optimizer_discriminator_logvar, mode='min', factor=0.3, patience=3)

모델의 테스트 결과를 t-sne 분포, boxplot, kdeplot으로 나타내었다.
