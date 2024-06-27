import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import random
from model_week4 import *
from rev_model_week4 import *
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

# 손실 함수 정의
def loss_function_manual(x_recon, x, mu, logvar):
    BCE_manu = torch.mean((x_recon-x)**2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())# see Appendix B from VAE paper:
                                                                 # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014

    return BCE_manu + 1.0*KLD + 1e-12

# 고유 파형들 정의
features=['A+IGBT-I', 'A+*IGBT-I', 'B+IGBT-I', 'B+*IGBT-I', 'C+IGBT-I', 
          'C+*IGBT-I', 'A-FLUX', 'B-FLUX', 'C-FLUX', 'CB-I','CB-V',
          'MOD-V', 'MOD-I','DV/DT']

system1='RFQ'     # pick a system to load and plot. Choose RFQ
system2='DTL'     # pick a system to load and plot. Choose DTL
system3='CCL'     # pick a system to load and plot. Choose CCL
system4='SCL'     # pick a system to load and plot. Choose SCL

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

a_FLUX_FAULT_INDICES = np.where(Yanomaly_RFQ[:, 2] == 'A FLUX Low Fault')[0] # A-Flux Low Fault만을 수집

data_test = Xnormal_concat[524*3:524*3+1,:,:] # visualize할 정상신호

# data_test = Xanomaly_RFQ[a_FLUX_FAULT_INDICES[5:],:,:] # visualize할 고장신호

# # 테스트 데이터셋 단일 파형(A-FLUX) min-max 스케일링 
# data_test = np.array(data_test)
# data_test = scaler_FLUX.fit_transform(data_test[:,:,0:1].reshape(-1,1)).reshape(data_test[:,:,0:1].shape)

# 테스트 데이터셋 min-max 스케일링
for i in range(len(features)):
    if i>=0 and i<=5:
        data_test[:,:,i] = scaler_IGBT.fit_transform(data_test[:,:,i].reshape(-1, 1)).reshape(data_test[:,:,i].shape)
    if i>=6 and i<=8:
        data_test[:,:,i] = scaler_FLUX.fit_transform(data_test[:,:,i].reshape(-1, 1)).reshape(data_test[:,:,i].shape)
    if i>=9 and i<=10:
        data_test[:,:,i] = scaler_CAP.fit_transform(data_test[:,:,i].reshape(-1, 1)).reshape(data_test[:,:,i].shape)
    if i>=11:
        data_test[:,:,i] = scaler_MOD.fit_transform((data_test[:,:,i]).reshape(-1, 1)).reshape((data_test[:,:,i]).shape)

data_test = data_test.transpose(0,2,1)
print(f'the shape of data_test : {data_test.shape}')
data_test = torch.tensor(data_test, dtype=torch.float32)
labels_one_hot = torch.tensor(labels_one_hot[524*3:524*3+1,:], dtype=torch.float32)

# RFQ 정상신호 평균값
avg_RFQ_Normal = np.zeros(Xnormal_CCL[:1,:,:].shape)
avg_RFQ_Normal = Xnormal_CCL[:1,:,:]
for i in range(1, 524):
    avg_RFQ_Normal += Xnormal_CCL[i:i+1,:,:]

for i in range(0, 14):
    avg_RFQ_Normal[:,:,i] = avg_RFQ_Normal[:,:,i] / 524

avg_RFQ_Normal_plot = avg_RFQ_Normal[0]
# GPU 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 및 학습 설정
input_dim = 14
hidden_dim = 4500
latent_dim = 512
condition_dim = 4
dropout=0.2

###### 4주차 베이스라인 모델 ######
# model = CVAE(
#         input_dim=input_dim, 
#         # seq_len=input_dim, 
#         latent_dim=latent_dim,
#         hidden_dim=hidden_dim, 
#         condition_dim=condition_dim,
#         dropout_prob=dropout
#         ).to(device).to(device)

###### maxpooling layer에 커널사이즈 변경 : 2->12 ###### 
model = CVAE_rev(
        input_dim=input_dim, 
        # seq_len=input_dim, 
        latent_dim=latent_dim,
        hidden_dim=hidden_dim, 
        condition_dim=condition_dim,
        dropout_prob=dropout
        ).to(device).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# 학습한 모델 불러오기
model.load_state_dict(torch.load('./model/CVAE_0627_SCL.pth'))

# 예측 및 결과 시각화
N = 0
model.eval()
with torch.no_grad():
    index = 524*0+N
    sample = data_test[index].unsqueeze(0).to(device)
    print(f'the shape of samples : {sample.shape}')

    print(f'the shape of one-hot labels : {labels_one_hot[index:index+1].shape}')

    condition = labels_one_hot[index].unsqueeze(0).to(device)
    print(f'the shape of condition : {condition.shape}')

    reconstructed, _, _ = model(sample, condition)

    # print(f'the shape of sample : {sample.shape}')
    # print(f'the shape of reconstructed : {reconstructed.shape}')

    # print(f'the shape of sample.squeeze(0) : {sample.squeeze(0).shape}')
    # print(f'the shape of reconstructed.squeeze(0) : {reconstructed.squeeze(0).shape}')       
    
    # print(sample.transpose(1,0)[0])
    # print(reconstructed.transpose(1,0)[0])

    # 원래 데이터 형태로 복원
    sample = sample.squeeze(0).cpu().numpy().transpose(1, 0)
    reconstructed = reconstructed.squeeze(0).cpu().numpy().transpose(1, 0)
    
    print(sample.shape)
    print(reconstructed.shape)

    # sample = scaler_FLUX.inverse_transform(sample.reshape(-1,1))
    # reconstructed = scaler_FLUX.inverse_transform(reconstructed.reshape(-1,1))

    # 역스케일링 추가
    for i in range(len(features)):
        if i>=0 and i<=5:
            sample[:,i] = scaler_IGBT.inverse_transform(sample[:,i].reshape(-1, 1)).squeeze()
            reconstructed[:,i] = scaler_IGBT.inverse_transform(reconstructed[:,i].reshape(-1, 1)).squeeze()
        if i>=6 and i<=8:
            sample[:,i] = scaler_FLUX.inverse_transform(sample[:,i].reshape(-1, 1)).squeeze()
            reconstructed[:,i] = scaler_FLUX.inverse_transform(reconstructed[:,i].reshape(-1, 1)).squeeze()
        if i>=9 and i<=10:
            sample[:,i] = scaler_CAP.inverse_transform(sample[:,i].reshape(-1, 1)).squeeze()
            reconstructed[:,i] = scaler_CAP.inverse_transform(reconstructed[:,i].reshape(-1, 1)).squeeze()
        if i>=11:
            sample[:,i] = scaler_MOD.inverse_transform(sample[:,i].reshape(-1, 1)).squeeze()
            reconstructed[:,i] = scaler_MOD.inverse_transform(reconstructed[:,i].reshape(-1, 1)).squeeze()

    print(sample)
    print(reconstructed)

# 단일 파형 plot하기
# for i in range(len(features)):
#     if i == 0:
#         plt.figure(figsize=(12, 6))
#         plt.plot(sample[:, i], label="Original")
#         plt.plot(reconstructed[:, i], label="Reconstructed")
#         plt.legend()
#         plt.xlabel("Time (s)")
#         plt.ylabel(features[i+6])
#         plt.title("Original vs Reconstructed")

#         # 특수문자들 저장할때 오류나서 제대로 되지 않기에, 다음 값들로 변환함.
#         # */ >> _
#         if features[i] == 'A+*IGBT-I':
#             features[i] = 'A+_IGBT-I'
#         if features[i] == 'B+*IGBT-I':
#             features[i] = 'B+_IGBT-I'
#         if features[i] == 'C+*IGBT-I':
#             features[i] = 'C+_IGBT-I'
#         if features[i] == 'DV/DT':
#             features[i] = 'DV_DT'
        
#         plt.savefig('./figure/0624/multi-A-FLUX_manual/' + str(features[i+6]) + '.png', dpi=600)

for i in range(len(features)):
    plt.figure(figsize=(12, 6))
    plt.plot(sample[:, i], label="Original")
    plt.plot(reconstructed[:, i], label="Reconstructed")
    plt.plot(avg_RFQ_Normal_plot[:, i], label="Average")
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel(features[i])
    plt.title("Original vs Reconstructed")

    # 특수문자들 저장할때 오류나서 제대로 되지 않기에, 다음 값들로 변환함.
    # */ >> _
    if features[i] == 'A+*IGBT-I':
        features[i] = 'A+_IGBT-I'
    if features[i] == 'B+*IGBT-I':
        features[i] = 'B+_IGBT-I'
    if features[i] == 'C+*IGBT-I':
        features[i] = 'C+_IGBT-I'
    if features[i] == 'DV/DT':
        features[i] = 'DV_DT'
    
    plt.savefig('./figure/0627/SCL/' + str(features[i]) + '.png', dpi=600)
