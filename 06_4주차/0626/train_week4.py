import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import random
from model_week4 import *

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

# 손실 함수 정의
def loss_function_lib(x_recon, x, mu, logvar):
    # print(x_recon.shape)
    # print(x.shape)
    
    BCE_lib = nn.functional.mse_loss(x_recon, x, reduction='sum')
    
    # print(x_recon - x)
    # print((x_recon - x)**2)
    # print(BCE_origin)
    # print(BCE_manu)
    
    # exit()
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())# see Appendix B from VAE paper:
                                                                 # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014

    return BCE_lib + 1.0*KLD + 1e-12

def loss_function_manual(x_recon, x, mu, logvar):
    
    BCE_manu = torch.mean((x_recon-x)**2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())# see Appendix B from VAE paper:
                                                                 # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014

    return BCE_manu + 1.0*KLD + 1e-12
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

standardized_IGBT = StandardScaler()
standardized_FLUX = StandardScaler()
standardized_CAP = StandardScaler()
sstandardized_MOD = StandardScaler()

Xnormal_concat = np.concatenate( (Xnormal_RFQ[:524,:,:], Xnormal_DTL[:524,:,:], Xnormal_CCL[:524,:,:], Xnormal_SCL[:524,:,:]), axis=0 )
Xnormal_concat = np.array(Xnormal_concat)

# 데이터 스케일링

# 정규화
# for i in range(len(features)):
#     if i<=5 :
#         Xnormal_concat[:,:,i] = standardized_IGBT.fit_transform(np.array(Xnormal_concat[:,:,i]).reshape(-1,1)).reshape(np.array(Xnormal_concat[:,:,i]).shape)
#     if i>=6 and i<=8 :
#         Xnormal_concat[:,:,i] = standardized_FLUX.fit_transform(Xnormal_concat[:,:,i].reshape(-1,1)).reshape(Xnormal_concat[:,:,i].shape)
#     if i>=9 and i<=10 :
#         Xnormal_concat[:,:,i] = standardized_CAP.fit_transform(Xnormal_concat[:,:,i].reshape(-1,1)).reshape(Xnormal_concat[:,:,i].shape)
#     if i>=11 :
#         Xnormal_concat[:,:,i] = sstandardized_MOD.fit_transform(Xnormal_concat[:,:,i].reshape(-1,1)).reshape(Xnormal_concat[:,:,i].shape)
# print(f'the shape of standardized signal in all unique waves(14) : {Xnormal_concat.shape}')

# # 정상신호 Min-Max 스케일링
# for i in range(len(features)):
#     if i<=5 or (i>=14 and i<=19):
#         Xnormal_concat[:,:,i] = scaler_IGBT.fit_transform(np.array(Xnormal_concat[:,:,i]).reshape(-1,1)).reshape(np.array(Xnormal_concat[:,:,i]).shape)
#     if (i>=6 and i<=8) or (i>=20 and i<=22):
#         Xnormal_concat[:,:,i] = scaler_FLUX.fit_transform(Xnormal_concat[:,:,i].reshape(-1,1)).reshape(Xnormal_concat[:,:,i].shape)
#     if (i>=9 and i<=10) or (i>=23 and i<=24):
#         Xnormal_concat[:,:,i] = scaler_CAP.fit_transform(Xnormal_concat[:,:,i].reshape(-1,1)).reshape(Xnormal_concat[:,:,i].shape)
#     if (i>=11 and i<=13) or (i>=25 and i<=27) :
#         Xnormal_concat[:,:,i] = scaler_MOD.fit_transform(Xnormal_concat[:,:,i].reshape(-1,1)).reshape(Xnormal_concat[:,:,i].shape)

print(f'the shape of standardized & min-max signal in all unique waves(14) : {Xnormal_concat.shape}')

# 정상신호 N개 학습
data1 = Xnormal_concat[524*0:524*0+524,:,:] # 4가지 모듈(RFQ, DTL, CCL, SCL) 정보를 가진524 * 4 = 2096개의 데이터를 학습함. 

# A-FLUX 오류신호 학습
# a_FLUX_FALUT_INDICES = np.where(Yanomaly_RFQ[:, 2] == 'A FLUX Low Fault')[0]
# # data1 = Xanomaly_RFQ[a_FLUX_FALUT_INDICES[:5],:,:]
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
# data1 = sclaer.fit_transform(data1.reshape(-1, data1.shape[-1])).reshape(data1.shape)

# 원-핫 인코딩
labels_concat = np.concatenate((labels1[normal_indices_RFQ[:524]], labels2[normal_indices_DTL[:524]], labels3[normal_indices_CCL[:524]], labels4[normal_indices_SCL[:524]]), axis=0)
labels_one_hot = np.eye(len(system_indices))[labels_concat]
labels_one_hot = torch.tensor(labels_one_hot, dtype=torch.float32) # to use the output of conditional VAE


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

num_epochs = 500
num_trials = 1   # default : 150
tmp = 100000000

# 모델 학습
for trial in range(num_trials):
    model_lib = CVAE(
        input_dim=input_dim, 
        latent_dim=latent_dim,
        hidden_dim=hidden_dim, 
        condition_dim=condition_dim,
        dropout_prob=dropout
        ).to(device)
    optimizer = optim.Adam(model_lib.parameters(), lr=1e-5) # 학습률 : 아담(adam) 옵티마이저 사용, 스케일 : 10^-5

    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)  # 학습률 스케쥴링

    total_loss_per_trial = 0
    model_lib.train()
    for epoch in range(num_epochs):
        total_loss = 0
        index = 1
        for (x, c) in dataloader1:
            x = x.to(device)
            c = c.to(device)
            optimizer.zero_grad()
            # 모델의 출력 : reconstruction_x, mean and log-variance used in the reconstruction of x
            x_recon_lib, mu, logvar = model_lib(x, c)

            # # 모델의 입력 및 출력의 shape 확인
            # print(x.shape)
            # print(x_recon_lib.shape)

            # 손실함수 계산 및 backpropagation
            loss = loss_function_lib(x_recon_lib, x, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            avg_loss = total_loss / (len(dataloader1)+1)

        total_loss_per_trial += avg_loss
        print(f'Trial {trial + 1}/{num_trials}, Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')
    
        
    print(f'Trial {trial + 1}/{num_trials} completed with total average loss: {total_loss_per_trial / num_epochs:.4f}')
print("라이브러리 모델 학습 완료!")
torch.save(model_lib.state_dict(), "./model/CVAE_0626_RFQ.pth")

exit()
# 모델 학습
for trial in range(num_trials):
    model_manual = CVAE(input_dim, hidden_dim, latent_dim, condition_dim).to(device)
    optimizer = optim.Adam(model_manual.parameters(), lr=1e-5) # 학습률 : 아담(adam) 옵티마이저 사용, 스케일 : 10^-5
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)  # 학습률 스케쥴링
    total_loss_per_trial = 0
    model_manual.train()
    for epoch in range(num_epochs):
        total_loss = 0
        index = 1
        for (x, c) in dataloader1:
            x = x.to(device)
            c = c.to(device)
            optimizer.zero_grad()
            x_recon_manual, mu, logvar = model_manual(x, c)
            # print(x_recon)
            loss = loss_function_manual(x_recon_manual, x, mu, logvar)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            avg_loss = total_loss / (len(dataloader1)+1)
        total_loss_per_trial += avg_loss
        print(f'Trial {trial + 1}/{num_trials}, Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')
    if avg_loss <= tmp :
        # 모델 저장
        tmp = avg_loss
        torch.save(model_manual.state_dict(), "./model/multi-module_based_cvae_loss_manual_only_RFQ.pth")
        print("최고 모델 저장 완료!")
        
    print(f'Trial {trial + 1}/{num_trials} completed with total average loss: {total_loss_per_trial / num_epochs:.4f}')
print("수동 MSE 손실함수 모델 학습 완료!")
# # 모델 저장
# torch.save(model.state_dict(), "./model/mutli-module_based_cvae_loss_tuning_aflux_backup.pth")

# print("모델 백업본 저장 완료!")
