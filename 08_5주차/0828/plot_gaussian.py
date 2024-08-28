import numpy as np
import matplotlib.pyplot as plt

###################### 제안한 방법을 사용하여 학습한 결과, 손실함수의 정상신호 및 사전오류신호 가우시안 그래프 그리기 ######################

# 정상신호의 평균과 표준 편차
mean_normal = 0.00109  # 평균 (μ)
std_dev_normal = 0.00050  # 표준 편차 (σ)

# 비정상신호의 평균과 표준 편차
mean_abnormal = 0.02082  # 평균 (μ)
std_dev_abnormal = 0.02795  # 표준 편차 (σ)

# x 좌표 생성 (평균을 중심으로 -3σ부터 +3σ까지의 범위)
x1 = np.linspace(mean_normal - 3*std_dev_normal, mean_normal + 3*std_dev_normal, 117*16)
x2 = np.linspace(mean_abnormal - 3*std_dev_abnormal, mean_abnormal + 3*std_dev_abnormal, 107*16)

# 가우시안 분포 함수 정의
def gaussian(x, mean, std_dev):
    return (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)

# y 좌표 생성 (확률 밀도 함수 값)
y1 = gaussian(x1, mean_normal, std_dev_normal)
y2 = gaussian(x2, mean_abnormal, std_dev_abnormal)

# 그래프 그리기
plt.figure(figsize=(8, 5))
plt.plot(x1, y1, label=f'μ = {mean_normal}, σ = {std_dev_normal}, normal')
plt.plot(x2, y2, label=f'μ = {mean_abnormal}, σ = {std_dev_abnormal}, abnormal')
plt.title('Gaussian Distribution')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.grid(True)
plt.legend()
plt.savefig('./gaussian_proposed.png',dpi=600)
