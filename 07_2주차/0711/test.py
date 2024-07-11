import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import random
import seaborn as sns
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
    BCE = nn.functional.mse_loss(x_recon, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())# see Appendix B from VAE paper:
                                                                 # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014

    return BCE + 1.0*KLD + 1e-12

# 고유 파형들 정의
features=['A+IGBT-I', 'A+*IGBT-I', 'B+IGBT-I', 'B+*IGBT-I', 'C+IGBT-I', 
          'C+*IGBT-I', 'A-FLUX', 'B-FLUX', 'C-FLUX', 'CB-I','CB-V',
          'MOD-V', 'MOD-I','DV/DT']

system1='RFQ'     # pick a system to load and plot. Choose RFQ
system2='DTL'     # pick a system to load and plot. Choose DTL
system3='CCL'     # pick a system to load and plot. Choose CCL
system4='SCL'     # pick a system to load and plot. Choose SCL
systems = ['RFQ','DTL','CCL','SCL']
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

# 정상,비정상 신호 concatenate
Xnormal_concat = np.concatenate( (Xnormal_RFQ, Xnormal_DTL, Xnormal_CCL, Xnormal_SCL), axis=0 )
Xanomaly_concat = np.concatenate( (Xanomaly_RFQ, Xanomaly_DTL, Xanomaly_CCL, Xanomaly_SCL), axis=0 )

# 원-핫 인코딩
labels_concat = np.concatenate((labels1[normal_indices_RFQ[:524]], labels2[normal_indices_DTL[:524]], labels3[normal_indices_CCL[:524]], labels4[normal_indices_SCL[:524]]), axis=0)
labels_one_hot = np.eye(len(system_indices))[labels_concat]

# Min-Max 스케일러- 파형별 ㄴ인스턴스 초기화
scaler_IGBT = MinMaxScaler()
scaler_FLUX = MinMaxScaler()
scaler_CAP = MinMaxScaler()
scaler_MOD = MinMaxScaler()

Xnormal_concat = np.concatenate( (Xnormal_RFQ[:524,:,:], Xnormal_DTL[:524,:,:], Xnormal_CCL[:524,:,:], Xnormal_SCL[:524,:,:]), axis=0 )
Xnormal_concat_tmp = np.array(Xnormal_concat)
Xnormal_concat = np.array(Xnormal_concat)

print(Xnormal_concat[524*0:524*0+1, :, 6]) # 오리지날 신호 확인

# 학습에 사용한 min-max 스케일러 객체들 정의.
for i in range(len(features)):
    if i>=0 and i<=5:
        Xnormal_concat_tmp[:,:,i] = scaler_IGBT.fit_transform(Xnormal_concat_tmp[:,:,i].reshape(-1, 1)).reshape(Xnormal_concat_tmp[:,:,i].shape)
    if i>=6 and i<=8:
        Xnormal_concat_tmp[:,:,i] = scaler_FLUX.fit_transform(Xnormal_concat_tmp[:,:,i].reshape(-1, 1)).reshape(Xnormal_concat_tmp[:,:,i].shape)
    if i>=9 and i<=10:
        Xnormal_concat_tmp[:,:,i] = scaler_CAP.fit_transform(Xnormal_concat_tmp[:,:,i].reshape(-1, 1)).reshape(Xnormal_concat_tmp[:,:,i].shape)
    if i>=11:
        Xnormal_concat_tmp[:,:,i] = scaler_MOD.fit_transform((Xnormal_concat_tmp[:,:,i]).reshape(-1, 1)).reshape((Xnormal_concat_tmp[:,:,i]).shape)

# print(f'the value of min-max signal : {Xnormal_concat_tmp[524*0:524*0+1, :, 6]}') # Min-Max 스케일링된 신호 확인

# Flux Low Fault
a_FLUX_FAULT_RFQ = np.where(Yanomaly_RFQ[:, 2] == 'A FLUX Low Fault')[0] # A-Flux Low Fault만을 수집
b_FLUX_FAULT_RFQ = np.where(Yanomaly_RFQ[:, 2] == 'B FLUX Low Fault')[0]
c_FLUX_FAULT_RFQ = np.where(Yanomaly_RFQ[:, 2] == 'C FLUX Low Fault')[0]
tmp_flux_RFQ = len(a_FLUX_FAULT_RFQ) + len(b_FLUX_FAULT_RFQ) + len(c_FLUX_FAULT_RFQ)

a_FLUX_FAULT_DTL = np.where(Yanomaly_DTL[:, 2] == 'A FLUX Low Fault')[0] # A-Flux Low Fault만을 수집
b_FLUX_FAULT_DTL = np.where(Yanomaly_DTL[:, 2] == 'B FLUX Low Fault')[0]
c_FLUX_FAULT_DTL = np.where(Yanomaly_DTL[:, 2] == 'C FLUX Low Fault')[0]
tmp_flux_DTL = len(a_FLUX_FAULT_DTL) + len(b_FLUX_FAULT_DTL) + len(c_FLUX_FAULT_DTL)

a_FLUX_FAULT_CCL = np.where(Yanomaly_CCL[:, 2] == 'A FLUX Low Fault')[0] # A-Flux Low Fault만을 수집
b_FLUX_FAULT_CCL = np.where(Yanomaly_CCL[:, 2] == 'B FLUX Low Fault')[0]
c_FLUX_FAULT_CCL = np.where(Yanomaly_CCL[:, 2] == 'C FLUX Low Fault')[0]
tmp_flux_CCL = len(a_FLUX_FAULT_CCL) + len(b_FLUX_FAULT_CCL) + len(c_FLUX_FAULT_CCL)

a_FLUX_FAULT_SCL = np.where(Yanomaly_SCL[:, 2] == 'A FLUX Low Fault')[0] # A-Flux Low Fault만을 수집
b_FLUX_FAULT_SCL = np.where(Yanomaly_SCL[:, 2] == 'B FLUX Low Fault')[0]
c_FLUX_FAULT_SCL = np.where(Yanomaly_SCL[:, 2] == 'C FLUX Low Fault')[0]
tmp_flux_SCL = len(a_FLUX_FAULT_SCL) + len(b_FLUX_FAULT_SCL) + len(c_FLUX_FAULT_SCL)

# dv_dt high/low fault
dv_dt_high_fault_RFQ = np.where(Yanomaly_RFQ[:, 2]== 'DV/DT High Fault')[0]
dv_dt_low_fault_RFQ = np.where(Yanomaly_RFQ[:, 2]== 'DV/DT Low Fault')[0]
tmp_RFQ = len(dv_dt_high_fault_RFQ) + len(dv_dt_low_fault_RFQ)

dv_dt_high_fault_DTL = np.where(Yanomaly_DTL[:, 2]== 'DV/DT High Fault')[0]
dv_dt_low_fault_DTL = np.where(Yanomaly_DTL[:, 2]== 'DV/DT Low Fault')[0]
tmp_DTL = len(dv_dt_high_fault_DTL) + len(dv_dt_low_fault_DTL)

dv_dt_high_fault_CCL = np.where(Yanomaly_CCL[:, 2]== 'DV/DT High Fault')[0]
dv_dt_low_fault_CCL = np.where(Yanomaly_CCL[:, 2]== 'DV/DT Low Fault')[0]
tmp_CCL = len(dv_dt_high_fault_CCL) + len(dv_dt_low_fault_CCL)

dv_dt_high_fault_SCL = np.where(Yanomaly_SCL[:, 2]== 'DV/DT High Fault')[0]
dv_dt_low_fault_SCL = np.where(Yanomaly_SCL[:, 2]== 'DV/DT Low Fault')[0]
tmp_SCL = len(dv_dt_high_fault_SCL) + len(dv_dt_low_fault_SCL)


data_test = Xnormal_concat[:,:,:] # visualize할 정상신호

# data_anomaly = np.concatenate(
#     (Xanomaly_RFQ[a_FLUX_FAULT_RFQ,:,:],Xanomaly_RFQ[b_FLUX_FAULT_RFQ,:,:],Xanomaly_RFQ[c_FLUX_FAULT_RFQ,:,:],
#      Xanomaly_DTL[a_FLUX_FAULT_DTL,:,:],Xanomaly_DTL[b_FLUX_FAULT_DTL,:,:],Xanomaly_DTL[c_FLUX_FAULT_DTL,:,:],
#      Xanomaly_CCL[a_FLUX_FAULT_CCL,:,:],Xanomaly_CCL[b_FLUX_FAULT_CCL,:,:],Xanomaly_CCL[c_FLUX_FAULT_CCL,:,:],
#      Xanomaly_SCL[a_FLUX_FAULT_SCL,:,:],Xanomaly_SCL[b_FLUX_FAULT_SCL,:,:],Xanomaly_SCL[c_FLUX_FAULT_SCL,:,:]), axis=0)

data_anomaly = np.concatenate(
    (Xanomaly_RFQ[dv_dt_high_fault_RFQ,:,:],Xanomaly_RFQ[dv_dt_low_fault_RFQ,:,:],
     Xanomaly_DTL[dv_dt_high_fault_DTL,:,:],Xanomaly_DTL[dv_dt_low_fault_DTL,:,:],
     Xanomaly_CCL[dv_dt_high_fault_CCL,:,:],Xanomaly_CCL[dv_dt_low_fault_CCL,:,:],
     Xanomaly_SCL[dv_dt_high_fault_SCL,:,:],Xanomaly_SCL[dv_dt_low_fault_SCL,:,:]),axis=0)

# 테스트 데이터셋 min-max 스케일링.
for i in range(len(features)):
    if i>=0 and i<=5:
        data_test[:,:,i] = scaler_IGBT.fit_transform(data_test[:,:,i].reshape(-1, 1)).reshape(data_test[:,:,i].shape)
        data_anomaly[:,:,i] = scaler_IGBT.fit_transform(data_anomaly[:,:,i].reshape(-1, 1)).reshape(data_anomaly[:,:,i].shape)
    if i>=6 and i<=8:
        data_test[:,:,i] = scaler_FLUX.fit_transform(data_test[:,:,i].reshape(-1, 1)).reshape(data_test[:,:,i].shape)
        data_anomaly[:,:,i] = scaler_FLUX.fit_transform(data_anomaly[:,:,i].reshape(-1, 1)).reshape(data_anomaly[:,:,i].shape)
    if i>=9 and i<=10:
        data_test[:,:,i] = scaler_CAP.fit_transform(data_test[:,:,i].reshape(-1, 1)).reshape(data_test[:,:,i].shape)
        data_anomaly[:,:,i] = scaler_CAP.fit_transform(data_anomaly[:,:,i].reshape(-1, 1)).reshape(data_anomaly[:,:,i].shape)
    if i>=11:
        data_test[:,:,i] = scaler_MOD.fit_transform((data_test[:,:,i]).reshape(-1, 1)).reshape((data_test[:,:,i]).shape)
        data_anomaly[:,:,i] = scaler_MOD.fit_transform((data_anomaly[:,:,i]).reshape(-1, 1)).reshape((data_anomaly[:,:,i]).shape)

data_test = data_test.transpose(0,2,1)
data_anomaly = data_anomaly.transpose(0,2,1)
print(f'the shape of data_test : {data_test.shape}')

data_test = torch.tensor(data_test, dtype=torch.float32)
data_anomaly = torch.tensor(data_anomaly, dtype=torch.float32)
# labels = torch.tensor(labels_one_hot[524*0:524*0+524,:], dtype=torch.float32)
labels = torch.tensor(labels_one_hot, dtype=torch.float32)

# GPU 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 및 학습 설정
input_dim = 14
hidden_dim = 4500
latent_dim = 512
condition_dim = 4
dropout=0.2

###### maxpooling layer에 커널사이즈 변경 : 2->12 ###### 
model = CVAE_rev(
        input_dim=input_dim, 
        # seq_len=input_dim, 
        latent_dim=latent_dim,
        hidden_dim=hidden_dim, 
        condition_dim=condition_dim,
        dropout_prob=dropout
        ).to(device).to(device)

# 최적화 함수 정의하기
optimizer = optim.Adam(model.parameters(), lr=1e-5)
# 학습한 모델 불러오기
model.load_state_dict(torch.load('./model/CVAE_0711_all.pth'))

# 모델 평가 및 결과 시각화
model.eval()
with torch.no_grad():
    mse = []
    mse_error = []
    labels_list = []
    data_test, labels = data_test.to(device), labels.to(device)
    x_recon, _, _ = model(data_test, labels)
    mse.extend(((data_test - x_recon) ** 2).mean(dim=(1,2)).cpu().numpy())
    # labels_list.extend(labels_one_hot.cpu().numpy())

# anomaly_labels = np.concatenate((labels_one_hot[:tmp_flux_RFQ,:],labels_one_hot[524*1:524*1+tmp_flux_DTL,:],labels_one_hot[524*2:524*2+tmp_flux_CCL,:],labels_one_hot[524*3:524*3+tmp_flux_SCL,:]), axis=0)
anomaly_labels = np.concatenate((labels_one_hot[:tmp_RFQ,:],labels_one_hot[524*1:524*1+tmp_DTL,:],labels_one_hot[524*2:524*2+tmp_CCL,:],labels_one_hot[524*3:524*3+tmp_SCL,:]), axis=0)
anomaly_labels = torch.tensor(anomaly_labels, dtype=torch.float32)


model.eval()
with torch.no_grad():
    mse_error = []
    labels_list = []
    data_anomaly, anomaly_labels = data_anomaly.to(device), anomaly_labels.to(device)
    x_error_recon, _, _ = model(data_anomaly, anomaly_labels)
    mse_error.extend(((data_anomaly - x_error_recon) ** 2).mean(dim=(1,2)).cpu().numpy())

mse = np.array(mse)
mse_error = np.array(mse_error)

# mse_scale = MinMaxScaler()
# mse_error_scale = MinMaxScaler()
# mse = np.array(mse_scale.fit_transform(np.array(mse).reshape(len(mse),1))).reshape(-1)
# mse_error = np.array(mse_error_scale.fit_transform((np.array(mse_error).reshape(len(mse_error),1)))).reshape(-1)

labels_list = np.array(labels_list)

# 이상 탐지 시각화

for i in range(0,1):
    # normal_mse = mse[labels_list == systems.index('RFQ')]

    plt.figure(figsize=(12, 6))
    # plt.hist(mse, bins=50, alpha=0.5, label='Normal data',color='#007FFF')
    # plt.hist(mse_error, bins=50, alpha=0.5, label='DV/DT Fault',color='#FF0000')
    sns.kdeplot((mse), fill=True,label='Normal',color='#007FFF')
    sns.kdeplot((mse_error), fill=True,label='DV/DT Fault',color='#FF0000')
    plt.legend()
    plt.xlabel('MSE')
    plt.ylabel('Density')
    plt.title(f'Normal in RFQ vs. DV/DT Fault in all modules')
    
    plt.savefig('./figure/0711/density/' + str('Normal vs. DV_DT Fault in all modules') + '.png', dpi=600)
