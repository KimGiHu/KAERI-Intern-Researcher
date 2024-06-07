import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
import random

# 시드 값 설정
seed = 42

# 기본 시드 고정
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# CUDA 사용 시 추가 설정
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 멀티-GPU 사용 시
    # CuDNN 결정론적 및 비결정론적 동작 설정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 데이터 준비
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.data[idx]

# VAE 모델 정의
class Encoder(nn.Module):
    def __init__(self, input_dim, seq_len, latent_dim, dropout_prob=0.2):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 128, kernel_size=12, padding=6)
        self.bn1 = nn.BatchNorm1d(num_features=128)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1_dropout = nn.Dropout(p=dropout_prob)

        self.conv2 = nn.Conv1d(128, 128, kernel_size=12, padding=6)
        self.bn2 = nn.BatchNorm1d(num_features=128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2_dropout = nn.Dropout(p=dropout_prob)

        self.conv3 = nn.Conv1d(128, 128, kernel_size=12, padding=6)
        self.bn3 = nn.BatchNorm1d(num_features=128)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3_dropout = nn.Dropout(p=dropout_prob)

        def calc_seq_len(seq_len):
            seq_len = (seq_len + 2 * 6 - 12) // 1 + 1  # conv1
            seq_len = (seq_len - 2) // 2 + 1           # pool1
            seq_len = (seq_len + 2 * 6 - 12) // 1 + 1  # conv2
            seq_len = (seq_len - 2) // 2 + 1           # pool2
            seq_len = (seq_len + 2 * 6 - 12) // 1 + 1  # conv3
            seq_len = (seq_len - 2) // 2 + 1           # pool3
            return seq_len

        self.seq_len = calc_seq_len(seq_len)
        self.fc = nn.Linear(128 * self.seq_len, 512)
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.bn0 = nn.BatchNorm1d(num_features=512)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.conv1_dropout(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        x = self.conv2_dropout(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.pool3(x)
        x = self.conv3_dropout(x)

        x = x.view(x.size(0), -1) 
        x = self.dropout(torch.relu(self.bn0(self.fc(x))))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, seq_len, output_dim, dropout_prob=0.2):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 512)
        self.fc2 = nn.Linear(512, 128 * seq_len)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.bn0 = nn.BatchNorm1d(num_features=128)

        self.up1 = nn.Upsample(scale_factor=2)
        self.bn1 = nn.BatchNorm1d(num_features=128)
        self.trans_conv1 = nn.ConvTranspose1d(128, 128, kernel_size=12, padding=6)
        self.conv1_dropout = nn.Dropout(p=dropout_prob)

        self.up2 = nn.Upsample(scale_factor=2)
        self.bn2 = nn.BatchNorm1d(num_features=128)
        self.trans_conv2 = nn.ConvTranspose1d(128, 128, kernel_size=12, padding=6)
        self.conv2_dropout = nn.Dropout(p=dropout_prob)

        self.up3 = nn.Upsample(size=4501)
        self.bn3 = nn.BatchNorm1d(num_features=128)
        self.trans_conv3 = nn.ConvTranspose1d(128, output_dim, kernel_size=12, padding=6)
        self.conv3_dropout = nn.Dropout(p=dropout_prob)

    def forward(self, z):
        h = self.dropout(torch.relu(self.fc1(z)))
        h = self.dropout(torch.relu(self.fc2(h)))
        h = h.view(h.size(0), 128, -1)
        h = self.bn0(h)
        h = self.up1(h)
        h = self.bn1(self.conv1_dropout(torch.relu(self.trans_conv1(h))))
        
        h = self.up2(h)
        h = self.bn2(self.conv2_dropout(torch.relu(self.trans_conv2(h))))
        
        h = self.up3(h)
        x_recon = self.conv3_dropout(torch.sigmoid(self.trans_conv3(h)))
        return x_recon
    
    
class VAE(nn.Module):
    def __init__(self, input_dim, seq_len, latent_dim, dropout_prob=0.2):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, seq_len, latent_dim, dropout_prob)
        self.decoder = Decoder(latent_dim, self.encoder.seq_len, input_dim, dropout_prob)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
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
X = np.load('./hvcm/data/hvcm/%s.npy' % system)
Y = np.load('./hvcm/data/hvcm/%s_labels.npy' % system, allow_pickle=True)
time = np.arange(X.shape[1]) * 400e-9 # 타임스텝 : 1.8ms (4500개 샘플씩, 한 샘플당 400ns)

# 배열 X,Y의 정상 및 오류 데이터들을 분리함
fault_indices, normal_indices = np.where(Y[:,1] == 'Fault')[0], np.where(Y[:,1] == 'Run')[0] 
Xnormal, Xanomaly = X[normal_indices,:,:], X[fault_indices,:,:]
Ynormal, Yanomaly = Y[normal_indices,:], Y[fault_indices,:]

# 데이터 생성 : A-Flux Low Fault만을 수집
a_FLUX_FAULT_INDICES = np.where(Yanomaly[:, 2] == 'A FLUX Low Fault')[0]

# 총 6개의 A-Flux Fault들 중에서 5개의 데이터를 이어 붙여서, 학습 데이터 셋으로 사용함.
data = Xanomaly[a_FLUX_FAULT_INDICES[:5],:,:]
data = MinMaxScaler().fit_transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)
data = data.transpose(0,2,1)  # (N,14,4500) 형태로 변경하여 Conv1D 입력 형식에 맞춤 
data = torch.tensor(data, dtype=torch.float32)

# 총 6개의 A-Flux Fault 중에서 마지막 인덱스 데이터를 테스트 셋으로 사용함.
data_test = Xanomaly[a_FLUX_FAULT_INDICES[-1:],:,:]
data_test = MinMaxScaler().fit_transform(data_test.reshape(-1, data_test.shape[-1])).reshape(data_test.shape)
data_test = data_test.transpose(0,2,1)
data_test = torch.tensor(data_test, dtype=torch.float32)

dataset = CustomDataset(data)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# GPU 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 및 학습 설정
input_dim = 14
hidden_dim = 4500
latent_dim = 512

model = VAE(input_dim, hidden_dim, latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-5)
num_epochs = 15000
num_trials = 1

# 모델 학습
for trial in range(num_trials):
    model = VAE(input_dim, hidden_dim, latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    total_loss_per_trial = 0
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for (x, _) in dataloader:
            x = x.to(device)
            optimizer.zero_grad()
            x_recon, mu, logvar = model(x)
            loss = loss_function(x_recon, x, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        total_loss_per_trial += avg_loss
        print(f'Trial {trial + 1}/{num_trials}, Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')
        
    print(f'Trial {trial + 1}/{num_trials} completed with total average loss: {total_loss_per_trial / num_epochs:.4f}')
print("모델 학습 완료!")

# 모델 저장
torch.save(model.state_dict(), "./model/vae_model_epoch15000_trial1.pth")

# 예측 및 결과 시각화
model.eval()
with torch.no_grad():
    sample = data_test[0].unsqueeze(0).to(device)
    reconstructed, _, _ = model(sample)

    sample = sample.squeeze().cpu().numpy().transpose(1, 0)  # 원래 데이터 형태로 복원
    reconstructed = reconstructed.squeeze().cpu().numpy().transpose(1, 0)  # 원래 데이터 형태로 복원

for i in range(len(features)):
    plt.figure(figsize=(12, 6))
    plt.plot(sample[:, i], label="Original")
    plt.plot(reconstructed[:, i], label="Reconstructed")
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel(features[i])
    plt.title("Original vs Reconstructed")
    if features[i] == 'DV/DT':
        features[i] = 'DV_DT'
    plt.savefig('./figure/epoch15000_trial1/' + str(features[i]) + '.png', dpi=600)
