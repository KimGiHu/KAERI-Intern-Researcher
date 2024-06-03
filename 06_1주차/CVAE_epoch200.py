import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler

# 데이터 준비
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.data[idx]

# CVAE 모델 정의
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, condition_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128 * hidden_dim, 512)
        self.fc_mu = nn.Linear(512 + condition_dim, latent_dim)
        self.fc_logvar = nn.Linear(512 + condition_dim, latent_dim)

    def forward(self, x, c):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc(x))
        x = torch.cat([x, c], dim=-1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, condition_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim + condition_dim, 512)
        self.fc2 = nn.Linear(512, 128 * hidden_dim)
        self.conv1 = nn.ConvTranspose1d(128, 128, kernel_size=3, padding=1)
        self.conv2 = nn.ConvTranspose1d(128, 128, kernel_size=3, padding=1)
        self.conv3 = nn.ConvTranspose1d(128, output_dim, kernel_size=3, padding=1)

    def forward(self, z, c):
        z = torch.cat([z, c], dim=-1)
        h = torch.relu(self.fc1(z))
        h = torch.relu(self.fc2(h))
        h = h.view(h.size(0), 128, -1)
        x_recon = torch.relu(self.conv1(h))
        x_recon = torch.relu(self.conv2(x_recon))
        x_recon = torch.sigmoid(self.conv3(x_recon))
        return x_recon

class CVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, condition_dim):
        super(CVAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, condition_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, condition_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c):
        mu, logvar = self.encoder(x, c)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z, c)
        return x_recon, mu, logvar

# 손실 함수 정의
def loss_function(x_recon, x, mu, logvar):
    BCE = nn.functional.mse_loss(x_recon, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# 고유 파형들 정의
features=['A+IGBT-I', 'A+*IGBT-I', 'B+IGBT-I', 'B+*IGBT-I', 'C+IGBT-I', 
          'C+*IGBT-I', 'A-FLUX', 'B-FLUX', 'C-FLUX', 'MOD-V', 'MOD-I', 'CB-I', 
          'CB-V', 'DV/DT']
feature_index=6  # A-flux 고유파형선택
system='RFQ'     # 시스템들( RFQ / DTL / CCL / SCL )  중 하나의 시스템을 선택함

# 파형 데이터셋 불러오기 (X : N_pulses, N_times, N_features) 과 (Y : N_pulses, N_labels - index, status, type)
# index : 파형의 path를 알려줌, status : 파형의 정상 또는 오류 여부, type : 정상이라면 Normal, 오류신호일 경우 종류를 표시 
X= np.load('./hvcm/data/hvcm/%s.npy' % system)
Y=np.load('./hvcm/data/hvcm/%s_labels.npy' % system, allow_pickle=True)
time=np.arange(X.shape[1]) * 400e-9 # 타임스텝 : 1.8ms (4500개 샘플씩, 한 샘플당 400ns)

# 배열 X,Y의 정상 및 오류 데이터들을 분리함
fault_indices, normal_indices = np.where(Y[:,1] == 'Fault')[0], np.where(Y[:,1] == 'Run')[0] 
Xnormal, Xanomaly = X[normal_indices,:,:], X[fault_indices,:,:]
Ynormal, Yanomaly = Y[normal_indices,:], Y[fault_indices,:]

# 데이터 생성 
a_FLUX_FALUT_INDICES = np.where(Yanomaly[:, 2] == 'A FLUX Low Fault')[0]
# print(a_FLUX_FALUT_INDICES)
# print(a_FLUX_FALUT_INDICES[:5])
# print(a_FLUX_FALUT_INDICES[-1:])

data = Xanomaly[a_FLUX_FALUT_INDICES[:5],:,:]
data = MinMaxScaler().fit_transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)
data = data.transpose(0,2,1)  # (N,14,4500) 형태로 변경하여 Conv1D 입력 형식에 맞춤 
data = torch.tensor(data, dtype=torch.float32)

data_test = Xanomaly[a_FLUX_FALUT_INDICES[-1:],:,:]

# print(data_test.shape)
# exit()

data_test = MinMaxScaler().fit_transform(data_test.reshape(-1, data_test.shape[-1])).reshape(data_test.shape)
data_test = data_test.transpose(0,2,1)
data_test = torch.tensor(data_test, dtype=torch.float32)
# labels=Yanomaly[a_FLUX_FALUT_INDICES,1].reshape(-1,1)


labels = torch.zeros((data.shape[0] ,1), dtype=torch.float32)
label_test = torch.zeros((data_test.shape[0], 1), dtype=torch.float32)

dataset = CustomDataset(data)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# GPU 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 및 학습 설정
input_dim = 14
hidden_dim = 4500
latent_dim = 512
condition_dim = 1

model = CVAE(input_dim, hidden_dim, latent_dim, condition_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-5)
num_epochs = 500

# 모델 학습
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for x, _ in dataloader:
        x = x.to(device)
        c = labels.to(device)
        optimizer.zero_grad()
        x_recon, mu, logvar = model(x, c)
        loss = loss_function(x_recon, x, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}')

print("모델 학습 완료!")

# 모델 저장
torch.save(model.state_dict(), "cvae_model.pth")

# exit()

# 모델 불러오기
# model = torch.load('./cvae_model.pth')
# 예측 및 결과 시각화
model.eval()
with torch.no_grad():
    sample = data_test[0].unsqueeze(0).to(device)
    condition = label_test[0].unsqueeze(0).to(device)
    reconstructed, _, _ = model(sample, condition)

    sample = sample.squeeze().cpu().numpy().transpose(1, 0)  # 원래 데이터 형태로 복원
    reconstructed = reconstructed.squeeze().cpu().numpy().transpose(1, 0)  # 원래 데이터 형태로 복원

for i in range(len(features)) :
    plt.figure(figsize=(12, 6))
    plt.plot(sample[:, i], label="Original")
    plt.plot(reconstructed[:, i], label="Reconstructed")
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel(features[i])
    plt.title("Original vs Reconstructed")
    plt.savefig('./figure/'+str(features[i])+'.png', dpi=600)
