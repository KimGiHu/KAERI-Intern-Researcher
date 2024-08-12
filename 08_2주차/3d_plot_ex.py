import numpy as np
import umap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 시계열 데이터 생성 (예시)
# (batch_size, time_steps, unique_features)
batch_size = 100
time_steps = 50
unique_features = 10

# 임의의 시계열 데이터 생성
data = np.random.rand(batch_size, time_steps, unique_features)

# 시간 지연 임베딩 함수
def time_delay_embedding(data, delay, dimension):
    n_samples, n_steps, n_features = data.shape
    embedded_data = np.zeros((n_samples, n_steps - (dimension - 1) * delay, dimension * n_features))
    for i in range(dimension):
        for j in range(n_features):
            embedded_data[:, :, i * n_features + j] = data[:, i * delay : n_steps - (dimension - 1 - i) * delay, j]
    return embedded_data

# 시간 지연 임베딩 적용
delay = 1
dimension = 5  # 임베딩 차원 설정
embedded_data = time_delay_embedding(data, delay, dimension)

# 임베딩된 데이터의 크기 확인
print("임베딩된 데이터의 크기:", embedded_data.shape)

# UMAP을 사용하여 차원 축소 (3차원)
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=3, metric='euclidean')
embedding = reducer.fit_transform(embedded_data.reshape(-1, embedded_data.shape[2]))

# 결과 시각화 (3D)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=np.arange(len(embedding)), cmap='viridis')
plt.colorbar(sc)
ax.set_title("3D UMAP Embedding of Time Series Data")
plt.savefig('example.png',dpi=600)
