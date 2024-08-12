import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import random
from july_4th_week.model import *


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
def loss_function(x_recon, x, mu, logvar):    
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

############################### 15가지 모듈의 정상신호 모음 ###############################
def return_indicies(data, module_name):
    indices = np.where(np.char.find(data, module_name) != -1)[0]
    return indices

def return_len(data, module_name):
    data_len = len(np.where(np.char.find(data, module_name) != -1)[0])
    return data_len

# 라벨링된 데이터 불러오기(?)
column_to_search_DTL_normal = Ynormal_DTL[:, 0].astype(str)
column_to_search_CCL_normal = Ynormal_CCL[:, 0].astype(str)
column_to_search_SCL_normal = Ynormal_SCL[:, 0].astype(str)

# RFQ 정상신호 
normal_RFQ = Xnormal_RFQ[:,:,:]

# DTL 정상신호 - DTL03, DTL05
normal_DTL3 = Xnormal_DTL[return_indicies(column_to_search_DTL_normal, 'DTL3'),:,:]
normal_DTL5 = Xnormal_DTL[return_indicies(column_to_search_DTL_normal, 'DTL5'),:,:]

# CCL 정상신호 - CCL01, CCL02, CCL03, CCL04
normal_CCL1 = Xnormal_CCL[return_indicies(column_to_search_CCL_normal, 'CCL1'),:,:]
normal_CCL2 = Xnormal_CCL[return_indicies(column_to_search_CCL_normal, 'CCL2'),:,:]
normal_CCL3 = Xnormal_CCL[return_indicies(column_to_search_CCL_normal, 'CCL3'),:,:]
normal_CCL4 = Xnormal_CCL[return_indicies(column_to_search_CCL_normal, 'CCL4'),:,:]

# SCL 정상신호 - SCL01, SCL05, SCL09, SCL12, SCL14, SCL15, SCL18, SCL21
normal_SCL1 = Xnormal_SCL[return_indicies(column_to_search_SCL_normal, 'SCL1'),:,:]
normal_SCL5 = Xnormal_SCL[return_indicies(column_to_search_SCL_normal, 'SCL5'),:,:]
normal_SCL9 = Xnormal_SCL[return_indicies(column_to_search_SCL_normal, 'SCL9'),:,:]
normal_SCL12 = Xnormal_SCL[return_indicies(column_to_search_SCL_normal, 'SCL12'),:,:]
normal_SCL14 = Xnormal_SCL[return_indicies(column_to_search_SCL_normal, 'SCL14'),:,:]
normal_SCL15 = Xnormal_SCL[return_indicies(column_to_search_SCL_normal, 'SCL15'),:,:]
normal_SCL18 = Xnormal_SCL[return_indicies(column_to_search_SCL_normal, 'SCL18'),:,:]
normal_SCL21 = Xnormal_SCL[return_indicies(column_to_search_SCL_normal, 'SCL21'),:,:]

# 데이터 분할 비율 설정
ratio = 0.6
ratio_target = 0.8

# Source Encoder Dataset
Xnormal_concat = np.concatenate( 
    (normal_RFQ[:int(len(normal_RFQ)*ratio),:,:],
     normal_DTL3[:int(return_len(column_to_search_DTL_normal,'DTL3')*ratio),:,:],
     normal_DTL5[:int(return_len(column_to_search_DTL_normal,'DTL5')*ratio),:,:],
     normal_CCL1[:int(return_len(column_to_search_CCL_normal,'CCL1')*ratio),:,:],
     normal_CCL2[:int(return_len(column_to_search_CCL_normal,'CCL2')*ratio),:,:],
     normal_CCL3[:int(return_len(column_to_search_CCL_normal,'CCL3')*ratio),:,:],
     normal_CCL4[:int(return_len(column_to_search_CCL_normal,'CCL4')*ratio),:,:],
     normal_SCL1[:int(return_len(column_to_search_SCL_normal,'SCL1')*ratio),:,:],
     normal_SCL5[:int(return_len(column_to_search_SCL_normal,'SCL5')*ratio),:,:],
     normal_SCL9[:int(return_len(column_to_search_SCL_normal,'SCL9')*ratio),:,:],
     normal_SCL12[:int(return_len(column_to_search_SCL_normal,'SCL12')*ratio),:,:],
     normal_SCL14[:int(return_len(column_to_search_SCL_normal,'SCL14')*ratio),:,:],
     normal_SCL15[:int(return_len(column_to_search_SCL_normal,'SCL15')*ratio),:,:],
     normal_SCL18[:int(return_len(column_to_search_SCL_normal,'SCL18')*ratio),:,:],
     normal_SCL21[:int(return_len(column_to_search_SCL_normal,'SCL21')*ratio),:,:]), axis=0 )

Xnormal_concat = np.array(Xnormal_concat)
data0 = Xnormal_concat[:,:,:]

# Target Encoder Dataset : anomaly dataset 60 percent
data1 = np.concatenate( 
    (normal_RFQ[int(len(normal_RFQ)*ratio):int(len(normal_RFQ)*ratio_target),:,:],
     normal_DTL3[int(return_len(column_to_search_DTL_normal,'DTL3')*ratio):int(return_len(column_to_search_DTL_normal,'DTL3')*ratio_target),:,:],
     normal_DTL5[int(return_len(column_to_search_DTL_normal,'DTL5')*ratio):int(return_len(column_to_search_DTL_normal,'DTL5')*ratio_target),:,:],
     normal_CCL1[int(return_len(column_to_search_CCL_normal,'CCL1')*ratio):int(return_len(column_to_search_CCL_normal,'CCL1')*ratio_target),:,:],
     normal_CCL2[int(return_len(column_to_search_CCL_normal,'CCL2')*ratio):int(return_len(column_to_search_CCL_normal,'CCL2')*ratio_target),:,:],
     normal_CCL3[int(return_len(column_to_search_CCL_normal,'CCL3')*ratio):int(return_len(column_to_search_CCL_normal,'CCL3')*ratio_target),:,:],
     normal_CCL4[int(return_len(column_to_search_CCL_normal,'CCL4')*ratio):int(return_len(column_to_search_CCL_normal,'CCL4')*ratio_target),:,:],
     normal_SCL1[int(return_len(column_to_search_SCL_normal,'SCL1')*ratio):int(return_len(column_to_search_SCL_normal,'SCL1')*ratio_target),:,:],
     normal_SCL5[int(return_len(column_to_search_SCL_normal,'SCL5')*ratio):int(return_len(column_to_search_SCL_normal,'SCL5')*ratio_target),:,:],
     normal_SCL9[int(return_len(column_to_search_SCL_normal,'SCL9')*ratio):int(return_len(column_to_search_SCL_normal,'SCL9')*ratio_target),:,:],
     normal_SCL12[int(return_len(column_to_search_SCL_normal,'SCL12')*ratio):int(return_len(column_to_search_SCL_normal,'SCL12')*ratio_target),:,:],
     normal_SCL14[int(return_len(column_to_search_SCL_normal,'SCL14')*ratio):int(return_len(column_to_search_SCL_normal,'SCL14')*ratio_target),:,:],
     normal_SCL15[int(return_len(column_to_search_SCL_normal,'SCL15')*ratio):int(return_len(column_to_search_SCL_normal,'SCL15')*ratio_target),:,:],
     normal_SCL18[int(return_len(column_to_search_SCL_normal,'SCL18')*ratio):int(return_len(column_to_search_SCL_normal,'SCL18')*ratio_target),:,:],
     normal_SCL21[int(return_len(column_to_search_SCL_normal,'SCL21')*ratio):int(return_len(column_to_search_SCL_normal,'SCL21')*ratio_target)-1,:,:]), axis=0 )

# 데이터 Min-Max 스케일링
for i in range(len(features)):
    if i<=5 or (i>=14 and i<=19):
        data0[:,:,i] = scaler_IGBT.fit_transform(np.array(data0[:,:,i]).reshape(-1,1)).reshape(np.array(data0[:,:,i]).shape)
        data1[:,:,i] = scaler_IGBT.fit_transform(np.array(data1[:,:,i]).reshape(-1,1)).reshape(np.array(data1[:,:,i]).shape)

    if (i>=6 and i<=8) or (i>=20 and i<=22):
        data0[:,:,i] = scaler_FLUX.fit_transform(data0[:,:,i].reshape(-1,1)).reshape(data0[:,:,i].shape)
        data1[:,:,i] = scaler_FLUX.fit_transform(data1[:,:,i].reshape(-1,1)).reshape(data1[:,:,i].shape)
        
    if (i>=9 and i<=10) or (i>=23 and i<=24):
        data0[:,:,i] = scaler_CAP.fit_transform(data0[:,:,i].reshape(-1,1)).reshape(data0[:,:,i].shape)
        data1[:,:,i] = scaler_CAP.fit_transform(data1[:,:,i].reshape(-1,1)).reshape(data1[:,:,i].shape)
        
    if (i>=11 and i<=13) or (i>=25 and i<=27) :
        data0[:,:,i] = scaler_MOD.fit_transform(data0[:,:,i].reshape(-1,1)).reshape(data0[:,:,i].shape)
        data1[:,:,i] = scaler_MOD.fit_transform(data1[:,:,i].reshape(-1,1)).reshape(data1[:,:,i].shape)

# source and target unique waveform slicing
index_slice_start = 13
index_slice_end = 14
data0 = data0[:,:,index_slice_start:index_slice_end]
data1 = data1[:,:,index_slice_start:index_slice_end]

source = data0.transpose(0,2,1)  # (N,14,4500) 형태로 변경하여 Conv1D 입력 형식에 맞춤 
target = data1.transpose(0,2,1)  # (N,14,4500) 형태로 변경하여 Conv1D 입력 형식에 맞춤\

tensor_source = torch.tensor(source, dtype=torch.float32)
tensor_target = torch.tensor(target, dtype=torch.float32)


# 15개의 모듈(RFQ, DTL, CCL, SCL) 원-핫 인코딩
source_labels_concat = np.concatenate((labels1[normal_indices_RFQ[:int(len(normal_RFQ)*ratio)]],
                                
                                labels2[return_indicies(column_to_search_DTL_normal, 'DTL3')[:int(return_len(column_to_search_DTL_normal,'DTL3')*ratio)]],
                                labels2[return_indicies(column_to_search_DTL_normal, 'DTL5')[:int(return_len(column_to_search_DTL_normal,'DTL5')*ratio)]], 

                                labels3[return_indicies(column_to_search_CCL_normal, 'CCL1')[:int(return_len(column_to_search_CCL_normal,'CCL1')*ratio)]],
                                labels3[return_indicies(column_to_search_CCL_normal, 'CCL2')[:int(return_len(column_to_search_CCL_normal,'CCL2')*ratio)]],
                                labels3[return_indicies(column_to_search_CCL_normal, 'CCL3')[:int(return_len(column_to_search_CCL_normal,'CCL3')*ratio)]],
                                labels3[return_indicies(column_to_search_CCL_normal, 'CCL4')[:int(return_len(column_to_search_CCL_normal,'CCL4')*ratio)]],

                                labels4[return_indicies(column_to_search_SCL_normal, 'SCL1')[:int(return_len(column_to_search_SCL_normal,'SCL1')*ratio)]],
                                labels4[return_indicies(column_to_search_SCL_normal, 'SCL5')[:int(return_len(column_to_search_SCL_normal,'SCL5')*ratio)]],
                                labels4[return_indicies(column_to_search_SCL_normal, 'SCL9')[:int(return_len(column_to_search_SCL_normal,'SCL9')*ratio)]],
                                labels4[return_indicies(column_to_search_SCL_normal, 'SCL12')[:int(return_len(column_to_search_SCL_normal,'SCL12')*ratio)]],
                                labels4[return_indicies(column_to_search_SCL_normal, 'SCL14')[:int(return_len(column_to_search_SCL_normal,'SCL14')*ratio)]],
                                labels4[return_indicies(column_to_search_SCL_normal, 'SCL15')[:int(return_len(column_to_search_SCL_normal,'SCL15')*ratio)]],
                                labels4[return_indicies(column_to_search_SCL_normal, 'SCL18')[:int(return_len(column_to_search_SCL_normal,'SCL18')*ratio)]],
                                labels4[return_indicies(column_to_search_SCL_normal, 'SCL21')[:int(return_len(column_to_search_SCL_normal,'SCL21')*ratio)]]), axis=0)

tmp_source = np.eye(15)[source_labels_concat]
labels_source = torch.tensor(tmp_source, dtype=torch.float32)

# 데이터 분할 비율 설정
ratio = 0.6
ratio_target = 0.8
target_labels_concat = np.concatenate((labels1[normal_indices_RFQ[int(len(normal_RFQ)*(ratio)):int(len(normal_RFQ)*(ratio_target))]],
                                
                                labels2[return_indicies(column_to_search_DTL_normal, 'DTL3')[int(return_len(column_to_search_DTL_normal,'DTL3')*(ratio)):int(return_len(column_to_search_DTL_normal,'DTL3')*(ratio_target))]],
                                labels2[return_indicies(column_to_search_DTL_normal, 'DTL5')[int(return_len(column_to_search_DTL_normal,'DTL5')*(ratio)):int(return_len(column_to_search_DTL_normal,'DTL5')*(ratio_target))]], 

                                labels3[return_indicies(column_to_search_CCL_normal, 'CCL1')[int(return_len(column_to_search_CCL_normal,'CCL1')*(ratio)):int(return_len(column_to_search_CCL_normal,'CCL1')*(ratio_target))]],
                                labels3[return_indicies(column_to_search_CCL_normal, 'CCL2')[int(return_len(column_to_search_CCL_normal,'CCL2')*(ratio)):int(return_len(column_to_search_CCL_normal,'CCL2')*(ratio_target))]],
                                labels3[return_indicies(column_to_search_CCL_normal, 'CCL3')[int(return_len(column_to_search_CCL_normal,'CCL3')*(ratio)):int(return_len(column_to_search_CCL_normal,'CCL3')*(ratio_target))]],
                                labels3[return_indicies(column_to_search_CCL_normal, 'CCL4')[int(return_len(column_to_search_CCL_normal,'CCL4')*(ratio)):int(return_len(column_to_search_CCL_normal,'CCL4')*(ratio_target))]],

                                labels4[return_indicies(column_to_search_SCL_normal, 'SCL1')[int(return_len(column_to_search_SCL_normal,'SCL1')*(ratio)):int(return_len(column_to_search_SCL_normal,'SCL1')*(ratio_target))]],
                                labels4[return_indicies(column_to_search_SCL_normal, 'SCL5')[int(return_len(column_to_search_SCL_normal,'SCL5')*(ratio)):int(return_len(column_to_search_SCL_normal,'SCL5')*(ratio_target))]],
                                labels4[return_indicies(column_to_search_SCL_normal, 'SCL9')[int(return_len(column_to_search_SCL_normal,'SCL9')*(ratio)):int(return_len(column_to_search_SCL_normal,'SCL9')*(ratio_target))]],
                                labels4[return_indicies(column_to_search_SCL_normal, 'SCL12')[int(return_len(column_to_search_SCL_normal,'SCL12')*(ratio)):int(return_len(column_to_search_SCL_normal,'SCL12')*(ratio_target))]],
                                labels4[return_indicies(column_to_search_SCL_normal, 'SCL14')[int(return_len(column_to_search_SCL_normal,'SCL14')*(ratio)):int(return_len(column_to_search_SCL_normal,'SCL14')*(ratio_target))]],
                                labels4[return_indicies(column_to_search_SCL_normal, 'SCL15')[int(return_len(column_to_search_SCL_normal,'SCL15')*(ratio)):int(return_len(column_to_search_SCL_normal,'SCL15')*(ratio_target))]],
                                labels4[return_indicies(column_to_search_SCL_normal, 'SCL18')[int(return_len(column_to_search_SCL_normal,'SCL18')*(ratio)):int(return_len(column_to_search_SCL_normal,'SCL18')*(ratio_target))]],
                                labels4[return_indicies(column_to_search_SCL_normal, 'SCL21')[int(return_len(column_to_search_SCL_normal,'SCL21')*(ratio)):int(return_len(column_to_search_SCL_normal,'SCL21')*(ratio_target))-1]]), axis=0)

tmp_target = np.eye(15)[target_labels_concat]
labels_target = torch.tensor(tmp_target, dtype=torch.float32)

# 데이터로더 정의
dataset_source = CustomDataset(tensor_source, labels_source)
dataset_target = CustomDataset(tensor_target, labels_target)
dataloader_source = DataLoader(dataset_source, batch_size=16, shuffle=True)
dataloader_target = DataLoader(dataset_target, batch_size=16, shuffle=True)

# GPU 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 및 학습파라미터 설정
input_dim = (index_slice_end - index_slice_start) # default : 14
hidden_dim = 4500 # default : 4500
latent_dim = 512
condition_dim = 15 # 총 15가지의 시스템 모듈 이름을 조건으로 사용.
dropout = 0.2

num_epochs = 100

# ADDA 모델 정의
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.discriminator(x)

# 모델 초기화
source_model = CVAE_rev(input_dim=input_dim, 
        # seq_len=input_dim, 
        latent_dim=latent_dim,
        hidden_dim=hidden_dim, 
        condition_dim=condition_dim,
        dropout_prob=dropout
        ).to(device)
source_model.load_state_dict(torch.load('./model/pretrained_dvdt.pth'))
target_encoder = Encoder_rev(input_dim, hidden_dim, latent_dim, condition_dim, dropout).to(device)
discriminator_mu = Discriminator().to(device)
discriminator_logvar = Discriminator().to(device)

# 손실 함수 및 옵티마이저
optimizer_target = optim.Adam(target_encoder.parameters(), lr=1e-5)
optimizer_discriminator_mu = optim.Adam(discriminator_mu.parameters(), lr=1e-3)
optimizer_discriminator_logvar = optim.Adam(discriminator_logvar.parameters(), lr=1e-3)


# Target encoder pretraining function
if __name__ =='__main__':
    # to train discriminator and target encoder
    target_encoder.train()
    discriminator_mu.train()
    discriminator_logvar.train()

    for epoch in range(num_epochs):
        total_loss_d = 0
        total_loss_g = 0
        iter = 0
        for (source_x,source_label), (target_x, target_label) in zip(dataloader_source, dataloader_target):
            iter += 1
            # to put the data on tensor
            source_x, source_label = source_x.to(device), source_label.to(device)
            target_x, target_label = target_x.to(device), target_label.to(device)
            optimizer_discriminator_mu.zero_grad()
            optimizer_discriminator_logvar.zero_grad()        
            
            source_mu, source_logvar = source_model.encoder(source_x, source_label)
            target_mu, target_logvar = target_encoder(target_x, target_label)
            
            # reconstruciton
            x_recon_source, _, _ = source_model(source_x, source_label)
            x_recon_target, _, _ = source_model(target_x, target_label)
            # mu one,zero matrix
            source_mu_labels = torch.ones(source_mu.size(0), 1).to(device)
            target_mu_labels = torch.zeros(target_mu.size(0), 1).to(device)
            # logvar one,zero matrix
            source_logvar_labels = torch.ones(source_logvar.size(0), 1).to(device)
            target_logvar_labels = torch.zeros(target_logvar.size(0), 1).to(device)
            
            source_mu_output = discriminator_mu(source_mu)
            target_mu_output = discriminator_mu(target_mu)

            source_logvar_output = discriminator_logvar(source_logvar)
            target_logvar_output = discriminator_logvar(target_logvar)    
            
            loss_discriminator = (nn.BCELoss()(source_mu_output, source_mu_labels) + nn.BCELoss()(target_mu_output, target_mu_labels) + \
                                 nn.BCELoss()(source_logvar_output, source_logvar_labels) + nn.BCELoss()(target_logvar_output, target_logvar_labels)) / 4
            loss_recon = (nn.MSELoss()(x_recon_source, source_x) + nn.MSELoss()(x_recon_target, target_x))

            all_loss = loss_recon + loss_discriminator
            all_loss.backward()
            optimizer_discriminator_mu.step()
            optimizer_discriminator_logvar.step()
            
            total_loss_d += all_loss.item()

            # Train target encoder
            optimizer_target.zero_grad()
            target_mu, target_logvar = target_encoder(target_x, target_label)
            x_recon_target, _, _ = source_model(target_x, target_label)
            # target_z = source_model.reparameterize(target_mu, target_logvar)
            # print(f'target label : {(torch.ones(target_z.size(0), 1)).shape}')
            # print(f'target value : {discriminator(target_z).shape}')

            loss_target = nn.MSELoss()(x_recon_target, target_x) + \
                         (nn.BCELoss()(discriminator_mu(target_mu), torch.ones(target_mu.size(0), 1).to(device)) + \
                          nn.BCELoss()(discriminator_logvar(target_logvar), torch.ones(target_logvar.size(0), 1).to(device)))/2 
            
            # loss_target = nn.BCELoss()(discriminator(loss_KLD(target_mu, target_logvar).reshape(1,1)),torch.ones(1,1).to(device))
            
            loss_target.backward()
            optimizer_target.step()
            
            total_loss_g += loss_target.item()

            avg_loss_d = total_loss_d / (len(dataloader_source)+1) # 데이터로더의 길이는 전체 샘플수 / 배치사이즈 한 값이다.
            avg_loss_g = total_loss_g / (len(dataloader_source)+1) # 데이터로더의 길이는 전체 샘플수 / 배치사이즈 한 값이다.

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss_d: {avg_loss_d:.4f}, Loss_g: {avg_loss_g:.4f}')  

    print("정상신호 식별 네트워크 학습완료!")
    torch.save(target_encoder.state_dict(), "./model/baseline/normal_dvdt.pth")
 
