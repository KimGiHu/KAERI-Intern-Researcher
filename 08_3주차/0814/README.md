성능에 전혀 개선되지 아니하여서, 모델의 학습률을 변경함. (10^-5 ---> 10^-3)

게다가, 학습률 자동 스케쥴러를 추가하여서 성능을 향상시키도록 함. (별로 향상되어 보이지는 않음.)

추가적으로 해야할 일 들.

학습의 성능을 객관적인 지표로 판단할 필요가 있음.

1. 재**구성 오차 (Reconstruction Error)**:

-  Normal과 Fault 데이터를 사용했을 때, 두 모델(기존모델, 제안한 모델) 각각의 재구성 오차 분포를 비교
- **평가 지표**:
    - MSE의 평균값, 분산
    - 두 모델 간의 MSE 차이를 통계적으로 검증하는 t-test 또는 Mann-Whitney U test 등의 통계적 검증 방법을 사용
    - Boxplot에서 각 데이터의 분포를 비교하여, MSE의 중앙값 및 IQR(Interquartile Range)을 통해 모델의 안정성을 평가

2. **잠재공간에서의 분리 (Latent Space Separation)**:

- t-SNE를 사용하여 시각화된 잠재공간에서 Normal과 Fault 데이터 간의 분리가 얼마나 잘 되는지를 평가
- 평가 지표:
      - 클러스터링 지표: 예를 들어, Silhouette Score, Davies-Bouldin Index, 혹은 Adjusted Rand Index (ARI) 같은 클러스터링 품질 평가 지표를 사용
      - 분포 차이: 잠재공간의 밀도 차이를 비교하기 위해 Kullback-Leibler Divergence (KL Divergence) 같은 지표를 사용
      - 시각적 평가: 잠재공간에서의 Normal과 Fault 데이터가 얼마나 잘 분리되었는지를 시각적으로 확인
  
3. **이상 탐지 성능 (Anomaly Detection Performance)**:

- 두 모델이 정상과 비정상 신호를 얼마나 잘 구분하는지를 평가하기 위해 ROC-AUC (Receiver Operating Characteristic - Area Under the Curve)와 같은 지표를 사용
- 평가 지표:
        - ROC Curve와 AUC 값 비교
        - Precision-Recall Curve와 F1 Score를 비교할 수 있습니다.

4. **모델의 복잡도 (Model Complexity)**:

- 모델의 파라미터 수나 계산 복잡도에 따라 실제 사용에서의 효율성을 평가
- 평가 지표:
        - 파라미터 수 비교
        - 추론 시간 (Inference Time) 비교
