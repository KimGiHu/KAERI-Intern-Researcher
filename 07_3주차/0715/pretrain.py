import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import random
from model import *

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
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 손실 함수 정의 - 라이브러리
def loss_function_lib(x_recon, x, mu, logvar):    
    MSE_lib = nn.functional.mse_loss(x_recon, x, reduction='sum') # divided into batch size
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())# see Appendix B from VAE paper:
                                                                 # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    return MSE_lib + 1.0*KLD + 1e-12

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
# scaler = MinMaxScaler()
scaler_IGBT = MinMaxScaler()
scaler_FLUX = MinMaxScaler()
scaler_CAP = MinMaxScaler()
scaler_MOD = MinMaxScaler()

Xnormal_concat = np.concatenate( (Xnormal_RFQ[:524,:,:], Xnormal_DTL[:524,:,:], Xnormal_CCL[:524,:,:], Xnormal_SCL[:524,:,:]), axis=0 )
Xnormal_concat = np.array(Xnormal_concat)


# 정상신호 N개 학습
data1 = Xnormal_concat[:,:,:] # 4가지 모듈(RFQ, DTL, CCL, SCL) 정보를 가진524 * 4 = 2096개의 데이터를 학습함. 


# 데이터 Min-Max 스케일링
for i in range(len(features)):
    if i<=5 or (i>=14 and i<=19):
        data1[:,:,i] = scaler_IGBT.fit_transform(np.array(data1[:,:,i]).reshape(-1,1)).reshape(np.array(data1[:,:,i]).shape)
    if (i>=6 and i<=8) or (i>=20 and i<=22):
        data1[:,:,i] = scaler_FLUX.fit_transform(data1[:,:,i].reshape(-1,1)).reshape(data1[:,:,i].shape)
    if (i>=9 and i<=10) or (i>=23 and i<=24):
        data1[:,:,i] = scaler_CAP.fit_transform(data1[:,:,i].reshape(-1,1)).reshape(data1[:,:,i].shape)
    if (i>=11 and i<=13) or (i>=25 and i<=27) :
        data1[:,:,i] = scaler_MOD.fit_transform(data1[:,:,i].reshape(-1,1)).reshape(data1[:,:,i].shape)

data1 = data1.transpose(0,2,1)  # (N,14,4500) 형태로 변경하여 Conv1D 입력 형식에 맞춤 
data1 = torch.tensor(data1, dtype=torch.float32)

# 원-핫 인코딩
labels_concat = np.concatenate((labels1[normal_indices_RFQ[:524]], labels2[normal_indices_DTL[:524]], labels3[normal_indices_CCL[:524]], labels4[normal_indices_SCL[:524]]), axis=0)
labels_one_hot = np.eye(len(system_indices))[labels_concat]
labels_one_hot = torch.tensor(labels_one_hot, dtype=torch.float32) # to use the output of conditional VAE
# labels_one_hot = torch.tensor(labels_one_hot[524*0:524*0+524], dtype=torch.float32) # to use the output of conditional VAE


dataset1 = CustomDataset(data1, labels_one_hot)
dataloader1 = DataLoader(dataset1, batch_size=16, shuffle=True)

# GPU 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 및 학습파라미터 설정
input_dim = 14
hidden_dim = 4500 # default : 4500
latent_dim = 512
condition_dim = 4 # RFQ(0), DTL(1), CCL(2), SCL(3) 총 4가지의 시스템 모듈 이름을 조건으로 사용.
dropout = 0.2

num_epochs = 100
num_trials = 1   # default : 150
tmp = 100000000

# 모델 초기화
source_encoder = CVAE_rev(
    input_dim=input_dim, 
    latent_dim=latent_dim,
    hidden_dim=hidden_dim, 
    condition_dim=condition_dim,
    dropout_prob=dropout
).to(device)


# 손실 함수 및 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer_source = optim.Adam(source_encoder.parameters(), lr=1e-5)

# Source encoder pretraining function
if __name__ =='__main__':
    source_encoder.train()
    for epoch in range(num_epochs):
        total_loss = 0
        middle_loss = 0
        for x, c in dataloader1:
            x = x.to(device)
            c = c.to(device)
            optimizer_source.zero_grad()
            for k in range(len(x)):
                x_recon, mu, logvar = source_encoder(x[k:k+1,:,:], c[k:k+1,:])
                loss = loss_function_lib(x_recon, x[k:k+1,:,:], mu, logvar)
                loss.backward()
                middle_loss += loss.item()
                optimizer_source.step()

            total_loss += middle_loss
            # total_loss += middle_loss / (16*4500)
                
                
            avg_loss = total_loss / (len(dataloader1)+1) # 데이터로더의 길이는 전체 샘플수 / 배치사이즈 한 값이다.

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')  
    print("사전 모델 학습 완료!")
    torch.save(source_encoder.state_dict(), "./model/CVAE_pretrained.pth")


