# train only A-FLUX waveform
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
import random

# 데이터 준비
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# CVAE 모델 정의
class Encoder(nn.Module):
    def __init__(self, input_dim, seq_len, latent_dim, condition_dim, dropout_prob=0.2):
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
        # self.fc = nn.Linear(128 * self.seq_len + condition_dim, 512)
        self.fc = nn.Linear(128 * self.seq_len, 512)
        self.fc_mu = nn.Linear(512 + condition_dim, latent_dim)
        self.fc_logvar = nn.Linear(512 + condition_dim, latent_dim)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.bn0 = nn.BatchNorm1d(num_features=(512 + condition_dim))

    def forward(self, x, c):
        # CNN Block1
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool1(x)
        x = torch.relu(x)
        x = self.conv1_dropout(x)
        
        # CNN Block2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = torch.relu(x)
        x = self.conv2_dropout(x)
        
        # CNN Block3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.pool3(x)
        x = torch.relu(x)
        x = self.conv3_dropout(x)
        
        # Flatten 효과
        x = x.view(x.size(0), -1) 
        x = self.fc(x)

        x = torch.cat([x, c], dim=-1) 
        
        x = self.bn0(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, seq_len, output_dim, condition_dim, dropout_prob=0.2):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim + condition_dim, 512)
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

    def forward(self, z, c):
        z = torch.cat([z, c], dim=-1)   
        
        h = ((self.fc1(z)))
        h = ((self.fc2(h)))
        h = h.view(h.size(0), 128, -1)
        
        h = self.bn0(h)
        h = self.dropout(torch.relu(h))
        
        h = self.up1(h)
        h = self.bn1(h)
        h = (self.conv1_dropout(torch.relu(self.trans_conv1(h))))
        
        h = self.up2(h)
        h = self.bn2(h)
        h = (self.conv2_dropout(torch.relu(self.trans_conv2(h))))
        
        h = self.up3(h)
        h = self.bn3(h)
        x_recon = self.conv3_dropout(torch.sigmoid(self.trans_conv3(h)))
        
        return x_recon
    
    
class CVAE(nn.Module):
    def __init__(self, input_dim, seq_len, latent_dim, condition_dim, dropout_prob=0.2):
        super(CVAE, self).__init__()
        self.encoder = Encoder(input_dim, seq_len, latent_dim, condition_dim, dropout_prob)
        self.decoder = Decoder(latent_dim, self.encoder.seq_len, input_dim, condition_dim, dropout_prob)

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
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())# see Appendix B from VAE paper:
                                                                 # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014

    return BCE + 1.0*KLD

# 고유 파형들 정의
features=['A+IGBT-I', 'A+*IGBT-I', 'B+IGBT-I', 'B+*IGBT-I', 'C+IGBT-I', 
          'C+*IGBT-I', 'A-FLUX', 'B-FLUX', 'C-FLUX', 'CB-I','CB-V',
          'MOD-V', 'MOD-I','DV/DT']

# 모듈별 변수들 정의
system1='RFQ'
system2='DTL'
system3='CCL'
system4='SCL'

# 데이터 로드 및 라벨 인코딩
def load_data(system):
    X = np.load(f'./hvcm/data/hvcm/{system}.npy')
    Y = np.load(f'./hvcm/data/hvcm/{system}_labels.npy', allow_pickle=True)
    return X, Y

X1, Y1 = load_data(system1)
X2, Y2 = load_data(system2)
X3, Y3 = load_data(system3)
X4, Y4 = load_data(system4)

time = np.arange(X1.shape[1]) * 400e-9 # 타임스텝 : 1.8ms (4500개 샘플씩, 한 샘플당 400ns)

# 시스템 인덱스 생성
system_indices = {
    system1: 0,
    system2: 1,
    system3: 2,
    system4: 3
}

def create_labels(Y, system):
    labels = np.array([system_indices[system]] * len(Y))
    return labels

labels1 = create_labels(Y1, system1)
labels2 = create_labels(Y2, system2)
labels3 = create_labels(Y3, system3)
labels4 = create_labels(Y4, system4)

# 배열 X,Y의 정상 및 오류 데이터들을 분리함
fault_indices_RFQ, normal_indices_RFQ = np.where(Y1[:,1] == 'Fault')[0], np.where(Y1[:,1] == 'Run')[0] 
fault_indices_DTL, normal_indices_DTL = np.where(Y2[:,1] == 'Fault')[0], np.where(Y2[:,1] == 'Run')[0]
fault_indices_CCL, normal_indices_CCL = np.where(Y3[:,1] == 'Fault')[0], np.where(Y3[:,1] == 'Run')[0]
fault_indices_SCL, normal_indices_SCL = np.where(Y4[:,1] == 'Fault')[0], np.where(Y4[:,1] == 'Run')[0]

Xnormal_RFQ, Xanomaly_RFQ = X1[normal_indices_RFQ,:,:], X1[fault_indices_RFQ,:,:]
Xnormal_DTL, Xanomaly_DTL = X2[normal_indices_DTL,:,:], X2[fault_indices_DTL,:,:]
Xnormal_CCL, Xanomaly_CCL = X3[normal_indices_CCL,:,:], X3[fault_indices_CCL,:,:]
Xnormal_SCL, Xanomaly_SCL = X4[normal_indices_SCL,:,:], X4[fault_indices_SCL,:,:]

Ynormal_RFQ, Yanomaly_RFQ = Y1[normal_indices_RFQ,:], Y1[fault_indices_RFQ,:]
Ynormal_DTL, Yanomaly_DTL = Y2[normal_indices_DTL,:], Y2[fault_indices_DTL,:]
Ynormal_CCL, Yanomaly_CCL = Y3[normal_indices_CCL,:], Y3[fault_indices_CCL,:]
Ynormal_SCL, Yanomaly_SCL = Y4[normal_indices_SCL,:], Y4[fault_indices_SCL,:]

# Min-Max 스케일러 인스턴스 초기화
scaler_IGBT = MinMaxScaler()
scaler_FLUX = MinMaxScaler()
scaler_CAP = MinMaxScaler()
scaler_MOD = MinMaxScaler()

Xnormal_concat = np.concatenate( (Xnormal_RFQ[:524,:,:], Xnormal_DTL[:524,:,:], Xnormal_CCL[:524,:,:], Xnormal_SCL[:524,:,:]), axis=0 )
Xnormal_concat = np.array(Xnormal_concat)

for i in range(len(features)):
    if i<=5 :
        Xnormal_concat[:,:,i] = scaler_IGBT.fit_transform(np.array(Xnormal_concat[:,:,i]).reshape(-1,1)).reshape(np.array(Xnormal_concat[:,:,i]).shape)
    if i>=6 and i<=8 :
        Xnormal_concat[:,:,i] = scaler_FLUX.fit_transform(Xnormal_concat[:,:,i].reshape(-1,1)).reshape(Xnormal_concat[:,:,i].shape)
    if i>=9 and i<=10 :
        Xnormal_concat[:,:,i] = scaler_CAP.fit_transform(Xnormal_concat[:,:,i].reshape(-1,1)).reshape(Xnormal_concat[:,:,i].shape)
    if i>=11 :
        Xnormal_concat[:,:,i] = scaler_MOD.fit_transform(Xnormal_concat[:,:,i].reshape(-1,1)).reshape(Xnormal_concat[:,:,i].shape)
print(f'the shape of Xnormal_concat in all unique waves(14) : {Xnormal_concat.shape}')

# 원-핫 인코딩
labels_concat = np.concatenate((labels1[normal_indices_RFQ[:524]], labels2[normal_indices_DTL[:524]], labels3[normal_indices_CCL[:524]], labels4[normal_indices_SCL[:524]]), axis=0)
labels_one_hot = np.eye(len(system_indices))[labels_concat]

data1 = Xnormal_concat[:,:,6:7] # 4가지 모듈(RFQ, DTL, CCL, SCL) 정보를 가진524 * 4 = 2096개의 데이터를 학습함. 
# data1 = sclaer.fit_transform(data1.reshape(-1, data1.shape[-1])).reshape(data1.shape)
print(f'the shape of Xnormal_concat_in_A_FLUX : {data1.shape}')
data1 = data1.transpose(0,2,1)  # (N,14,4500) 형태로 변경하여 Conv1D 입력 형식에 맞춤 
data1 = torch.tensor(data1, dtype=torch.float32)
labels_one_hot = torch.tensor(labels_one_hot, dtype=torch.float32)

dataset1 = CustomDataset(data1, labels_one_hot)
dataloader1 = DataLoader(dataset1, batch_size=16, shuffle=True)

# GPU 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 및 학습파라미터 설정
input_dim = 1
hidden_dim = 4500
latent_dim = 512
condition_dim = 4 # RFQ(0), DTL(1), CCL(2), SCL(3) 총 4가지의 시스템 모듈 이름을 조건으로 사용.

num_epochs = 500 
num_trials = 1   # default : 150
tmp = 100000000

# 모델 학습
for trial in range(num_trials):
    model = CVAE(input_dim, hidden_dim, latent_dim, condition_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5) # 학습률 : 아담(adam) 옵티마이저 사용, 스케일 : 10^-5
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)  # 학습률 스케쥴링
    total_loss_per_trial = 0
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        index = 1
        for (x, c) in dataloader1:
            x = x.to(device)
            c = c.to(device)
            optimizer.zero_grad()
            x_recon, mu, logvar = model(x, c)
            loss = loss_function(x_recon, x, mu, logvar)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            avg_loss = total_loss / (len(dataloader1)+1)
        total_loss_per_trial += avg_loss
        print(f'Trial {trial + 1}/{num_trials}, Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')
    if avg_loss <= tmp :
        # 모델 저장
        tmp = avg_loss
        torch.save(model.state_dict(), "./model/mutli-module_based_cvae_only_A_FLUX.pth")
        print("최고 모델 저장 완료!")
        
    print(f'Trial {trial + 1}/{num_trials} completed with total average loss: {total_loss_per_trial / num_epochs:.4f}')
print("모델 학습 완료!")

# 모델 저장
# torch.save(model.state_dict(), "./model/mutli-module_based_cvae_best_ver_uniquewave_scaler_backup.pth")

print("모델 백업본 저장 완료!")
