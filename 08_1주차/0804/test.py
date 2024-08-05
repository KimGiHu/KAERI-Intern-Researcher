import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import random
import seaborn as sns
import argparse
from july_4th_week.model import *

print(torch.cuda.is_available())
# cuda 캐시 정리
torch.cuda.empty_cache()

# 오류 타입 실행할 때, 받아오기
parser = argparse.ArgumentParser()
parser.add_argument('--fault', type=str,
                    default='FLUX', help='set the type of fault')
args = parser.parse_args()

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

############################## 15가지 모듈의 이상신호 모음 ###############################
column_to_search_RFQ = Yanomaly_RFQ[:, 0].astype(str)
column_to_search_DTL = Yanomaly_DTL[:, 0].astype(str)
column_to_search_CCL = Yanomaly_CCL[:, 0].astype(str)
column_to_search_SCL = Yanomaly_SCL[:, 0].astype(str)

# RFQ 이상신호
abnormal_RFQ = Xanomaly_RFQ[:,:,:]

# DTL 이상신호 - DTL03, DTL05
abnormal_DTL3 = Xanomaly_DTL[return_indicies(column_to_search_DTL, 'DTL3'),:,:]
abnormal_DTL5 = Xanomaly_DTL[return_indicies(column_to_search_DTL, 'DTL5'),:,:]

# CCL 이상신호 - CCL01, CCL02, CCL03, CCL04
abnormal_CCL1 = Xanomaly_CCL[return_indicies(column_to_search_CCL, 'CCL1'),:,:]
abnormal_CCL2 = Xanomaly_CCL[return_indicies(column_to_search_CCL, 'CCL2'),:,:]
abnormal_CCL3 = Xanomaly_CCL[return_indicies(column_to_search_CCL, 'CCL3'),:,:]
abnormal_CCL4 = Xanomaly_CCL[return_indicies(column_to_search_CCL, 'CCL4'),:,:]

# SCL 이상신호 - SCL01, SCL05, SCL09, SCL12, SCL14, SCL15, SCL18, SCL21
abnormal_SCL1 = Xanomaly_SCL[return_indicies(column_to_search_SCL, 'SCL1'),:,:]
abnormal_SCL5 = Xanomaly_SCL[return_indicies(column_to_search_SCL, 'SCL5'),:,:]
abnormal_SCL9 = Xanomaly_SCL[return_indicies(column_to_search_SCL, 'SCL9'),:,:]
abnormal_SCL12 = Xanomaly_SCL[return_indicies(column_to_search_SCL, 'SCL12'),:,:]
abnormal_SCL14 = Xanomaly_SCL[return_indicies(column_to_search_SCL, 'SCL14'),:,:]
abnormal_SCL15 = Xanomaly_SCL[return_indicies(column_to_search_SCL, 'SCL15'),:,:]
abnormal_SCL18 = Xanomaly_SCL[return_indicies(column_to_search_SCL, 'SCL18'),:,:]
abnormal_SCL21 = Xanomaly_SCL[return_indicies(column_to_search_SCL, 'SCL21'),:,:]

# 테스트 : normal / abnormal 설정
ratio_target = 0.8
ratio_test = 1.0
data_test = np.concatenate( 
    (normal_RFQ[int(len(normal_RFQ)*ratio_target):int(len(normal_RFQ)*ratio_test),:,:],
     normal_DTL3[int(return_len(column_to_search_DTL_normal,'DTL3')*ratio_target):int(return_len(column_to_search_DTL_normal,'DTL3')*ratio_test),:,:],
     normal_DTL5[int(return_len(column_to_search_DTL_normal,'DTL5')*ratio_target):int(return_len(column_to_search_DTL_normal,'DTL5')*ratio_test),:,:],
     normal_CCL1[int(return_len(column_to_search_CCL_normal,'CCL1')*ratio_target):int(return_len(column_to_search_CCL_normal,'CCL1')*ratio_test),:,:],
     normal_CCL2[int(return_len(column_to_search_CCL_normal,'CCL2')*ratio_target):int(return_len(column_to_search_CCL_normal,'CCL2')*ratio_test),:,:],
     normal_CCL3[int(return_len(column_to_search_CCL_normal,'CCL3')*ratio_target):int(return_len(column_to_search_CCL_normal,'CCL3')*ratio_test),:,:],
     normal_CCL4[int(return_len(column_to_search_CCL_normal,'CCL4')*ratio_target):int(return_len(column_to_search_CCL_normal,'CCL4')*ratio_test),:,:],
     normal_SCL1[int(return_len(column_to_search_SCL_normal,'SCL1')*ratio_target):int(return_len(column_to_search_SCL_normal,'SCL1')*ratio_test),:,:],
     normal_SCL5[int(return_len(column_to_search_SCL_normal,'SCL5')*ratio_target):int(return_len(column_to_search_SCL_normal,'SCL5')*ratio_test),:,:],
     normal_SCL9[int(return_len(column_to_search_SCL_normal,'SCL9')*ratio_target):int(return_len(column_to_search_SCL_normal,'SCL9')*ratio_test),:,:],
     normal_SCL12[int(return_len(column_to_search_SCL_normal,'SCL12')*ratio_target):int(return_len(column_to_search_SCL_normal,'SCL12')*ratio_test),:,:],
     normal_SCL14[int(return_len(column_to_search_SCL_normal,'SCL14')*ratio_target):int(return_len(column_to_search_SCL_normal,'SCL14')*ratio_test),:,:],
     normal_SCL15[int(return_len(column_to_search_SCL_normal,'SCL15')*ratio_target):int(return_len(column_to_search_SCL_normal,'SCL15')*ratio_test),:,:],
     normal_SCL18[int(return_len(column_to_search_SCL_normal,'SCL18')*ratio_target):int(return_len(column_to_search_SCL_normal,'SCL18')*ratio_test),:,:],
     normal_SCL21[int(return_len(column_to_search_SCL_normal,'SCL21')*ratio_target):int(return_len(column_to_search_SCL_normal,'SCL21')*ratio_test)+1,:,:]), axis=0 )

fault_ratio = 0.6
fault_end_ratio = 1.0
data_anomaly = np.concatenate( 
    (abnormal_RFQ[int(len(abnormal_RFQ)*fault_ratio):int(len(abnormal_RFQ)*fault_end_ratio),:,:],
     abnormal_DTL3[int(return_len(column_to_search_DTL,'DTL3')*fault_ratio):int(return_len(column_to_search_DTL,'DTL3')*fault_end_ratio),:,:],
     abnormal_DTL5[int(return_len(column_to_search_DTL,'DTL5')*fault_ratio):int(return_len(column_to_search_DTL,'DTL5')*fault_end_ratio),:,:],
     abnormal_CCL1[int(return_len(column_to_search_CCL,'CCL1')*fault_ratio):int(return_len(column_to_search_CCL,'CCL1')*fault_end_ratio),:,:],
     abnormal_CCL2[int(return_len(column_to_search_CCL,'CCL2')*fault_ratio):int(return_len(column_to_search_CCL,'CCL2')*fault_end_ratio),:,:],
     abnormal_CCL3[int(return_len(column_to_search_CCL,'CCL3')*fault_ratio):int(return_len(column_to_search_CCL,'CCL3')*fault_end_ratio),:,:],
     abnormal_CCL4[int(return_len(column_to_search_CCL,'CCL4')*fault_ratio):int(return_len(column_to_search_CCL,'CCL4')*fault_end_ratio),:,:],
     abnormal_SCL1[int(return_len(column_to_search_SCL,'SCL1')*fault_ratio):int(return_len(column_to_search_SCL,'SCL1')*fault_end_ratio),:,:],
     abnormal_SCL5[int(return_len(column_to_search_SCL,'SCL5')*fault_ratio):int(return_len(column_to_search_SCL,'SCL5')*fault_end_ratio),:,:],
     abnormal_SCL9[int(return_len(column_to_search_SCL,'SCL9')*fault_ratio):int(return_len(column_to_search_SCL,'SCL9')*fault_end_ratio),:,:],
     abnormal_SCL12[int(return_len(column_to_search_SCL,'SCL12')*fault_ratio):int(return_len(column_to_search_SCL,'SCL12')*fault_end_ratio),:,:],
     abnormal_SCL14[int(return_len(column_to_search_SCL,'SCL14')*fault_ratio):int(return_len(column_to_search_SCL,'SCL14')*fault_end_ratio),:,:],
     abnormal_SCL15[int(return_len(column_to_search_SCL,'SCL15')*fault_ratio):int(return_len(column_to_search_SCL,'SCL15')*fault_end_ratio),:,:],
     abnormal_SCL18[int(return_len(column_to_search_SCL,'SCL18')*fault_ratio):int(return_len(column_to_search_SCL,'SCL18')*fault_end_ratio),:,:],
     abnormal_SCL21[int(return_len(column_to_search_SCL,'SCL21')*fault_ratio):int(return_len(column_to_search_SCL,'SCL21')*fault_end_ratio),:,:]), axis=0 )

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

# source and target unique waveform slicing
index_slice_start = 0
index_slice_end = 6
data_test = data_test[:,:,index_slice_start:index_slice_end]
data_anomaly = data_anomaly[:,:,index_slice_start:index_slice_end]

# 테스트셋 전치(transpose)
data_test = data_test.transpose(0,2,1)
data_anomaly = data_anomaly.transpose(0,2,1)
# 테스트셋 tensor위에 두기
data_test = torch.tensor(data_test, dtype=torch.float32)
data_anomaly = torch.tensor(data_anomaly, dtype=torch.float32)

# one-hot encoding label
normal_labels = np.concatenate((labels1[normal_indices_RFQ[int(len(normal_RFQ)*(ratio_target)):int(len(normal_RFQ)*(ratio_test))]],                           
                                labels2[return_indicies(column_to_search_DTL_normal, 'DTL3')[int(return_len(column_to_search_DTL_normal,'DTL3')*(ratio_target)):int(return_len(column_to_search_DTL_normal,'DTL3')*(ratio_test))]],
                                labels2[return_indicies(column_to_search_DTL_normal, 'DTL5')[int(return_len(column_to_search_DTL_normal,'DTL5')*(ratio_target)):int(return_len(column_to_search_DTL_normal,'DTL5')*(ratio_test))]], 
                                labels3[return_indicies(column_to_search_CCL_normal, 'CCL1')[int(return_len(column_to_search_CCL_normal,'CCL1')*(ratio_target)):int(return_len(column_to_search_CCL_normal,'CCL1')*(ratio_test))]],
                                labels3[return_indicies(column_to_search_CCL_normal, 'CCL2')[int(return_len(column_to_search_CCL_normal,'CCL2')*(ratio_target)):int(return_len(column_to_search_CCL_normal,'CCL2')*(ratio_test))]],
                                labels3[return_indicies(column_to_search_CCL_normal, 'CCL3')[int(return_len(column_to_search_CCL_normal,'CCL3')*(ratio_target)):int(return_len(column_to_search_CCL_normal,'CCL3')*(ratio_test))]],
                                labels3[return_indicies(column_to_search_CCL_normal, 'CCL4')[int(return_len(column_to_search_CCL_normal,'CCL4')*(ratio_target)):int(return_len(column_to_search_CCL_normal,'CCL4')*(ratio_test))]],
                                labels4[return_indicies(column_to_search_SCL_normal, 'SCL1')[int(return_len(column_to_search_SCL_normal,'SCL1')*(ratio_target)):int(return_len(column_to_search_SCL_normal,'SCL1')*(ratio_test))]],
                                labels4[return_indicies(column_to_search_SCL_normal, 'SCL5')[int(return_len(column_to_search_SCL_normal,'SCL5')*(ratio_target)):int(return_len(column_to_search_SCL_normal,'SCL5')*(ratio_test))]],
                                labels4[return_indicies(column_to_search_SCL_normal, 'SCL9')[int(return_len(column_to_search_SCL_normal,'SCL9')*(ratio_target)):int(return_len(column_to_search_SCL_normal,'SCL9')*(ratio_test))]],
                                labels4[return_indicies(column_to_search_SCL_normal, 'SCL12')[int(return_len(column_to_search_SCL_normal,'SCL12')*(ratio_target)):int(return_len(column_to_search_SCL_normal,'SCL12')*(ratio_test))]],
                                labels4[return_indicies(column_to_search_SCL_normal, 'SCL14')[int(return_len(column_to_search_SCL_normal,'SCL14')*(ratio_target)):int(return_len(column_to_search_SCL_normal,'SCL14')*(ratio_test))]],
                                labels4[return_indicies(column_to_search_SCL_normal, 'SCL15')[int(return_len(column_to_search_SCL_normal,'SCL15')*(ratio_target)):int(return_len(column_to_search_SCL_normal,'SCL15')*(ratio_test))]],
                                labels4[return_indicies(column_to_search_SCL_normal, 'SCL18')[int(return_len(column_to_search_SCL_normal,'SCL18')*(ratio_target)):int(return_len(column_to_search_SCL_normal,'SCL18')*(ratio_test))]],
                                labels4[return_indicies(column_to_search_SCL_normal, 'SCL21')[int(return_len(column_to_search_SCL_normal,'SCL21')*(ratio_target)):int(return_len(column_to_search_SCL_normal,'SCL21')*(ratio_test))+1]]), axis=0)

tmp_normal_labels = np.eye(15)[normal_labels]
labels = torch.tensor(tmp_normal_labels, dtype=torch.float32)

anomaly_labels = np.concatenate((labels1[fault_indices_RFQ[int(len(abnormal_RFQ)*fault_ratio):int(len(abnormal_RFQ)*fault_end_ratio)]],
                                labels2[return_indicies(column_to_search_DTL, 'DTL3')[int(return_len(column_to_search_DTL,'DTL3')*fault_ratio):int(return_len(column_to_search_DTL, 'DTL3')*fault_end_ratio)]],
                                labels2[return_indicies(column_to_search_DTL, 'DTL5')[int(return_len(column_to_search_DTL,'DTL5')*fault_ratio):int(return_len(column_to_search_DTL, 'DTL5')*fault_end_ratio)]], 
                                labels3[return_indicies(column_to_search_CCL, 'CCL1')[int(return_len(column_to_search_CCL,'CCL1')*fault_ratio):int(return_len(column_to_search_CCL, 'CCL1')*fault_end_ratio)]],
                                labels3[return_indicies(column_to_search_CCL, 'CCL2')[int(return_len(column_to_search_CCL,'CCL2')*fault_ratio):int(return_len(column_to_search_CCL, 'CCL2')*fault_end_ratio)]],
                                labels3[return_indicies(column_to_search_CCL, 'CCL3')[int(return_len(column_to_search_CCL,'CCL3')*fault_ratio):int(return_len(column_to_search_CCL, 'CCL3')*fault_end_ratio)]],
                                labels3[return_indicies(column_to_search_CCL, 'CCL4')[int(return_len(column_to_search_CCL,'CCL4')*fault_ratio):int(return_len(column_to_search_CCL, 'CCL4')*fault_end_ratio)]],
                                labels4[return_indicies(column_to_search_SCL, 'SCL1')[int(return_len(column_to_search_SCL,'SCL1')*fault_ratio):int(return_len(column_to_search_SCL, 'SCL1')*fault_end_ratio)]],
                                labels4[return_indicies(column_to_search_SCL, 'SCL5')[int(return_len(column_to_search_SCL,'SCL5')*fault_ratio):int(return_len(column_to_search_SCL, 'SCL5')*fault_end_ratio)]],
                                labels4[return_indicies(column_to_search_SCL, 'SCL9')[int(return_len(column_to_search_SCL,'SCL9')*fault_ratio):int(return_len(column_to_search_SCL, 'SCL9')*fault_end_ratio)]],
                                labels4[return_indicies(column_to_search_SCL, 'SCL12')[int(return_len(column_to_search_SCL,'SCL12')*fault_ratio):int(return_len(column_to_search_SCL, 'SCL12')*fault_end_ratio)]],
                                labels4[return_indicies(column_to_search_SCL, 'SCL14')[int(return_len(column_to_search_SCL,'SCL14')*fault_ratio):int(return_len(column_to_search_SCL, 'SCL14')*fault_end_ratio)]],
                                labels4[return_indicies(column_to_search_SCL, 'SCL15')[int(return_len(column_to_search_SCL,'SCL15')*fault_ratio):int(return_len(column_to_search_SCL, 'SCL15')*fault_end_ratio)]],
                                labels4[return_indicies(column_to_search_SCL, 'SCL18')[int(return_len(column_to_search_SCL,'SCL18')*fault_ratio):int(return_len(column_to_search_SCL, 'SCL18')*fault_end_ratio)]],
                                labels4[return_indicies(column_to_search_SCL, 'SCL21')[int(return_len(column_to_search_SCL,'SCL21')*fault_ratio):int(return_len(column_to_search_SCL, 'SCL21')*fault_end_ratio)]]), axis=0)

tmp_fault = np.eye(15)[anomaly_labels]
anomaly_labels = torch.tensor(tmp_fault, dtype=torch.float32)

# GPU 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# 모델 및 학습파라미터 설정
input_dim = (index_slice_end-index_slice_start) # default : 14
hidden_dim = 4500 # default : 4500
latent_dim = 512
condition_dim = 15 # RFQ(0), DTL(1), CCL(2), SCL(3) 총 4가지의 시스템 모듈 이름을 조건으로 사용.
dropout = 0.2
num_epochs = 100

# 모델 초기화
source_model = CVAE_rev(input_dim=input_dim, 
        # seq_len=input_dim, 
        latent_dim=latent_dim,
        hidden_dim=hidden_dim, 
        condition_dim=condition_dim,
        dropout_prob=dropout
        ).to(device)
state_dict = (torch.load('./model/IGBT_loss_tunning.pth'))
decoder_state_dict = {k[len("decoder."):]: v for k, v in state_dict.items() if k.startswith("decoder.")} # 디코더의 state_dict 키 필터링

# 정상신호 추론모델
class Valid_model_Normal(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, condition_dim, dropout_prob=0.2):
        super(Valid_model_Normal, self).__init__()
        self.encoder = Encoder_rev(input_dim, hidden_dim, latent_dim, condition_dim, dropout_prob)
        self.decoder = Decoder_rev(latent_dim, self.encoder.seq_len, input_dim, condition_dim, dropout_prob)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar + 1e-12)
        eps = torch.randn_like(std)
        return mu + eps * std + 1e-12

    def forward(self, x, c):
        self.encoder.load_state_dict(torch.load('./model/IGBT_abnormal.pth'))
        self.decoder.load_state_dict(decoder_state_dict)
        mu, logvar = self.encoder(x, c)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z, c)
        return x_recon, mu, logvar

# 비정상신호 추론모델
class Valid_model_Fault(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, condition_dim, dropout_prob=0.2):
        super(Valid_model_Fault, self).__init__()
        self.encoder = Encoder_rev(input_dim, hidden_dim, latent_dim, condition_dim, dropout_prob)
        self.decoder = Decoder_rev(latent_dim, self.encoder.seq_len, input_dim, condition_dim, dropout_prob)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar + 1e-12)
        eps = torch.randn_like(std)
        return mu + eps * std + 1e-12

    def forward(self, x, c):
        self.encoder.load_state_dict(torch.load('./model/IGBT_abnormal.pth'))
        self.decoder.load_state_dict(decoder_state_dict)
        mu, logvar = self.encoder(x, c)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z, c)
        return x_recon, mu, logvar
    
test_normal = Valid_model_Normal(
        input_dim=input_dim, 
        # seq_len=input_dim, 
        latent_dim=latent_dim,
        hidden_dim=hidden_dim, 
        condition_dim=condition_dim,
        dropout_prob=dropout
        ).to(device)

test_fault = Valid_model_Fault(
        input_dim=input_dim, 
        # seq_len=input_dim, 
        latent_dim=latent_dim,
        hidden_dim=hidden_dim, 
        condition_dim=condition_dim,
        dropout_prob=dropout
        ).to(device)
pretrain_model = CVAE_rev(
        input_dim=input_dim, 
        latent_dim=latent_dim,
        hidden_dim=hidden_dim, 
        condition_dim=condition_dim,
        dropout_prob=dropout
        ).to(device)


pretrain_model.load_state_dict(torch.load('./model/IGBT_loss_tunning.pth'))

# 손실 함수 및 옵티마이저
optimizer_normal = optim.Adam(test_normal.parameters(), lr=1e-5)
optimizer_fault = optim.Adam(test_fault.parameters(), lr=1e-5)
optimizer_pretrain = optim.Adam(pretrain_model.parameters(), lr=1e-5)

# Target encoder pretraining function

# to train discriminator
pretrain_model.eval()
test_normal.eval()
with torch.no_grad():
    mse = []
    mse_pretrain = []
    labels_list = []
    data_test, labels = data_test.to(device), labels.to(device)
    
    # 기존 모델
    x_recon_pretrain, _, _ = pretrain_model(data_test, labels)

    # 제안한 모델 
    x_recon_1, _, _ = test_normal(data_test, labels)
    # x_recon_2, _, _ = test_fault(data_test, labels)
    # x_recon = (x_recon_1 + x_recon_2) / 2

    # print(x_recon.shape)
    mse_pretrain.extend((((data_test - x_recon_pretrain) ** 2)).mean(dim=(1,2)).cpu().numpy()) # 기존모델을 이용해 복원한 정상신호의 MSE
                                                                                                                     # [:,13:14,:] : (배치사이즈, 고유파형의 인덱스, 샘플 수 : 4500개)
    mse.extend((((data_test - x_recon_1) ** 2)).mean(dim=(1,2)).cpu().numpy()) # 제안된 모델을 이용해 복원한 정상신호의 MSE
    # print((np.array(mse)).shape)

with torch.no_grad():
    mse_error = []
    mse_error_pretrain = []
    labels_list = []
    data_anomaly, anomaly_labels = data_anomaly.to(device), anomaly_labels.to(device)
    
    # 기존 모델
    x_pretrain_error_recon, _, _ = pretrain_model(data_anomaly, anomaly_labels)

    # 제안한 모델
    x_error_recon_1, _, _ = test_normal(data_anomaly, anomaly_labels)
    # x_error_recon_2, _, _ = test_fault(data_anomaly, anomaly_labels)
    # x_error_recon = (x_error_recon_1 + x_error_recon_2) / 2

    mse_error_pretrain.extend((((data_anomaly - x_pretrain_error_recon) ** 2)).mean(dim=(1,2)).cpu().numpy()) # 기존모델을 이용해 복원한 오류신호의 MSE
                                                                                                                                    # [:,13:14,:] : (배치사이즈, 고유파형의 인덱스, 샘플 수 : 4500개)
    mse_error.extend((((data_anomaly - x_error_recon_1) ** 2)).mean(dim=(1,2)).cpu().numpy()) # 제안된 모델을 이용해 복원한 오류신호의 MSE

mse = np.array(mse)
mse_pretrain = np.array(mse_pretrain)
mse_error = np.array(mse_error)
mse_error_pretrain = np.array(mse_error_pretrain)

labels_list = np.array(labels_list)

# LaTeX 스타일 활성화
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

import pandas as pd
data_pretrain_normal = pd.DataFrame({'Type':'pretrained Normal data','MSE_pretrain':mse_pretrain})
data_pretrain_fault = pd.DataFrame({'MSE_pretrain_error':mse_error_pretrain,'Type':'pretrained fault data'})
data_normal = pd.DataFrame({'MSE_normal':mse,'Type':'Normal data'})
data_fault = pd.DataFrame({'MSE_error':mse_error,'Type':'Fault data'})

# 이상 탐지 시각화
for i in range(0,1):
    plt.figure(figsize=(8, 6))

    # histogram
    # plt.hist(mse_pretrain, bins=100, alpha=0.5, label='pretrained Normal data') # 기존 모델을 이용해 복원한 정상신호 손실함수 분포
    # plt.hist(mse_error_pretrain, bins=100, alpha=0.5, label='pretrained Fault') # 기존 모델을 이용해 복원한 오류신호 손실함수 분포
    # plt.hist(mse, bins=100, alpha=0.5, label='Normal Loss',color='gray') # 제안된 모델을 이용해 복원한 정상신호 손실함수 분포
    # plt.hist(mse_error, bins=100, alpha=0.5, label='Fault Loss',color='red') # 제안된 모델을 이용해 복원한 정상신호 손실함수 분포
    
    # kernel distribution error
    sns.kdeplot((mse_pretrain), fill=True,label='pretrained Normal Loss',cut=0) # 기존 모델을 이용해 복원한 정상신호 손실함수 분포
    sns.kdeplot((mse_error_pretrain), fill=True,label='pretrained Fault Loss',cut=0) # 기존 모델을 이용해 복원한 오류신호 손실함수 분포
    sns.kdeplot((mse), fill=True,label='Normal Los',color='gray',cut=0) # 제안된 모델을 이용해 복원한 정상신호 손실함수 분포
    sns.kdeplot((mse_error), fill=True,label='Fault Loss',color='red',cut=0) # 제안된 모델을 이용해 복원한 정상신호 손실함수 분포

    # box plot : 
    # sns.boxplot(x='Type', y='MSE_pretrain', data = data_pretrain_normal, label='pretrained Normal') # 기존 모델을 이용해 복원한 정상신호 손실함수 분포
    # sns.boxplot(x='Type', y='MSE_pretrain_error', data = data_pretrain_fault, label='pretrained Fault') # 기존 모델을 이용해 복원한 오류신호 손실함수 분포
    # sns.boxplot(x='Type', y='MSE_normal', data = data_normal, label='proposed Normal') # 제안된 모델을 이용해 복원한 정상신호 손실함수 분포
    # sns.boxplot(x='Type', y='MSE_error', data = data_fault, label='proposed Fault') # 제안된 모델을 이용해 복원한 정상신호 손실함수 분포

    # sci 논문 양식
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5) # grid 설정
    plt.legend(fontsize=12)
    
    # plt.title(f'Normal vs. Fault ver.pretrain')
    plt.title(f'%s Normal vs. Fault box plot'%args.fault)
     
    # histogram & kdeplot settings
    plt.xlabel('MSE', fontsize=14, fontweight='bold')
    plt.ylabel('Loss Density', fontsize=14, fontweight='bold')
    plt.xscale('log')
    plt.xlim(1e-4, 1)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'$10^{{{int(np.log10(x))}}}$'))
    
    # box plot settings
    # plt.xlabel('Type', fontsize=14, fontweight='bold')
    # plt.ylabel('MSE', fontsize=14, fontweight='bold')
    # plt.yscale('log')
    # plt.ylim(1e-4, 1)
    # plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'$10^{{{int(np.log10(y))}}}$'))

    if f'{args.fault}' == 'DV/DT' :
        args.fault = 'DV_DT'

    # plt.savefig('./figure/0719/' + str('Normal vs. %s'%args.fault+' Fault') + '.png', dpi=600)
    # plt.savefig('./figure/0722/' + str('Normal vs. Fault ver.pretrain') + '.png', dpi=600)
    plt.savefig('./figure/0804/' +'%s'%args.fault + str(' Normal vs. Fault box plot') + '.png', dpi=600)
    print("테스트 결과 저장완료 !")
