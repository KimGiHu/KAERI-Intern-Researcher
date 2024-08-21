adaptation 모델에도 학습률 스케쥴러를 다음과 같이 이용함.

### 학습률 스케쥴러 추가
> scheduler_g = optim.lr_scheduler.ReduceLROnPlateau(optimizer_target, mode='min', factor=0.3, patience=3)
> scheduler_d_mu = optim.lr_scheduler.ReduceLROnPlateau(optimizer_discriminator_logvar, mode='min', factor=0.3, patience=3)
> scheduler_d_logvar = optim.lr_scheduler.ReduceLROnPlateau(optimizer_discriminator_logvar, mode='min', factor=0.3, patience=3)

모델의 테스트 결과를 t-sne 분포, boxplot, kdeplot으로 나타내었다.

### pretraine_ver2.py 설명

기존의 모델은 활성화 함수로 ReLU를 사용하였다. 하지만, 성능이 일정이상 오르지 않는 saturation현상이 발생하는 것을 확인하여, GELU 함수로 변경하여 실험을 진행해보았다. 그 결과는 디렉토리 gelu에 저장되어있다.

ver1 최종 average Loss : 0.2045
ver2 최종 average Loss : 0.2036
