import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score, confusion_matrix
import random
import seaborn as sns
import argparse
from model import *

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
ratio_target = 0.85
ratio_test = 1.0
data_test = np.concatenate( 
    (normal_RFQ[int(len(normal_RFQ)*ratio_target):int(len(normal_RFQ)*ratio_test),:,:],
     # DTL module
     normal_DTL3[int(return_len(column_to_search_DTL_normal,'DTL3')*ratio_target):int(return_len(column_to_search_DTL_normal,'DTL3')*ratio_test),:,:],
     normal_DTL5[int(return_len(column_to_search_DTL_normal,'DTL5')*ratio_target):int(return_len(column_to_search_DTL_normal,'DTL5')*ratio_test),:,:],
     # CCL module
     normal_CCL1[int(return_len(column_to_search_CCL_normal,'CCL1')*ratio_target):int(return_len(column_to_search_CCL_normal,'CCL1')*ratio_test),:,:],
     normal_CCL2[int(return_len(column_to_search_CCL_normal,'CCL2')*ratio_target):int(return_len(column_to_search_CCL_normal,'CCL2')*ratio_test),:,:],
     normal_CCL3[int(return_len(column_to_search_CCL_normal,'CCL3')*ratio_target):int(return_len(column_to_search_CCL_normal,'CCL3')*ratio_test),:,:],
     normal_CCL4[int(return_len(column_to_search_CCL_normal,'CCL4')*ratio_target):int(return_len(column_to_search_CCL_normal,'CCL4')*ratio_test),:,:],
     # SCL module
     normal_SCL1[int(return_len(column_to_search_SCL_normal,'SCL1')*ratio_target):int(return_len(column_to_search_SCL_normal,'SCL1')*ratio_test),:,:],
     normal_SCL5[int(return_len(column_to_search_SCL_normal,'SCL5')*ratio_target):int(return_len(column_to_search_SCL_normal,'SCL5')*ratio_test),:,:],
     normal_SCL9[int(return_len(column_to_search_SCL_normal,'SCL9')*ratio_target):int(return_len(column_to_search_SCL_normal,'SCL9')*ratio_test),:,:],
     normal_SCL12[int(return_len(column_to_search_SCL_normal,'SCL12')*ratio_target):int(return_len(column_to_search_SCL_normal,'SCL12')*ratio_test),:,:],
     normal_SCL14[int(return_len(column_to_search_SCL_normal,'SCL14')*ratio_target):int(return_len(column_to_search_SCL_normal,'SCL14')*ratio_test),:,:],
     normal_SCL15[int(return_len(column_to_search_SCL_normal,'SCL15')*ratio_target):int(return_len(column_to_search_SCL_normal,'SCL15')*ratio_test),:,:],
     normal_SCL18[int(return_len(column_to_search_SCL_normal,'SCL18')*ratio_target):int(return_len(column_to_search_SCL_normal,'SCL18')*ratio_test),:,:],
     normal_SCL21[int(return_len(column_to_search_SCL_normal,'SCL21')*ratio_target):int(return_len(column_to_search_SCL_normal,'SCL21')*ratio_test)+1,:,:]), axis=0 )

fault_ratio = 0.0
fault_end_ratio = 1.0
data_anomaly = np.concatenate( 
    (abnormal_RFQ[int(len(abnormal_RFQ)*fault_ratio):int(len(abnormal_RFQ)*fault_end_ratio),:,:],
     # DTL module
     abnormal_DTL3[int(return_len(column_to_search_DTL,'DTL3')*fault_ratio):int(return_len(column_to_search_DTL,'DTL3')*fault_end_ratio),:,:],
     abnormal_DTL5[int(return_len(column_to_search_DTL,'DTL5')*fault_ratio):int(return_len(column_to_search_DTL,'DTL5')*fault_end_ratio),:,:],
     # CCL module
     abnormal_CCL1[int(return_len(column_to_search_CCL,'CCL1')*fault_ratio):int(return_len(column_to_search_CCL,'CCL1')*fault_end_ratio),:,:],
     abnormal_CCL2[int(return_len(column_to_search_CCL,'CCL2')*fault_ratio):int(return_len(column_to_search_CCL,'CCL2')*fault_end_ratio),:,:],
     abnormal_CCL3[int(return_len(column_to_search_CCL,'CCL3')*fault_ratio):int(return_len(column_to_search_CCL,'CCL3')*fault_end_ratio),:,:],
     abnormal_CCL4[int(return_len(column_to_search_CCL,'CCL4')*fault_ratio):int(return_len(column_to_search_CCL,'CCL4')*fault_end_ratio),:,:],
     # SCL module
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
index_slice_end = 14
data_test = data_test[:,:,index_slice_start:index_slice_end]
data_anomaly = data_anomaly[:,:,index_slice_start:index_slice_end]

# 테스트셋 전치(transpose)
data_test = data_test.transpose(0,2,1)
data_anomaly = data_anomaly.transpose(0,2,1)
# 테스트셋 tensor위에 두기
data_test = torch.tensor(data_test, dtype=torch.float32)
data_anomaly = torch.tensor(data_anomaly, dtype=torch.float32)

# one-hot encoding label
normal_labels = np.concatenate((
    # RFQ module
    labels1[normal_indices_RFQ[int(len(normal_RFQ)*(ratio_target)):int(len(normal_RFQ)*(ratio_test))]],
    # DTL module
    labels2[return_indicies(column_to_search_DTL_normal, 'DTL3')[int(return_len(column_to_search_DTL_normal,'DTL3')*(ratio_target)):int(return_len(column_to_search_DTL_normal,'DTL3')*(ratio_test))]],
    labels2[return_indicies(column_to_search_DTL_normal, 'DTL5')[int(return_len(column_to_search_DTL_normal,'DTL5')*(ratio_target)):int(return_len(column_to_search_DTL_normal,'DTL5')*(ratio_test))]], 
    # CCL module
    labels3[return_indicies(column_to_search_CCL_normal, 'CCL1')[int(return_len(column_to_search_CCL_normal,'CCL1')*(ratio_target)):int(return_len(column_to_search_CCL_normal,'CCL1')*(ratio_test))]],
    labels3[return_indicies(column_to_search_CCL_normal, 'CCL2')[int(return_len(column_to_search_CCL_normal,'CCL2')*(ratio_target)):int(return_len(column_to_search_CCL_normal,'CCL2')*(ratio_test))]],
    labels3[return_indicies(column_to_search_CCL_normal, 'CCL3')[int(return_len(column_to_search_CCL_normal,'CCL3')*(ratio_target)):int(return_len(column_to_search_CCL_normal,'CCL3')*(ratio_test))]],
    labels3[return_indicies(column_to_search_CCL_normal, 'CCL4')[int(return_len(column_to_search_CCL_normal,'CCL4')*(ratio_target)):int(return_len(column_to_search_CCL_normal,'CCL4')*(ratio_test))]],
    # SCL module
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

anomaly_labels = np.concatenate((
    # RFQ module
    labels1[fault_indices_RFQ[int(len(abnormal_RFQ)*fault_ratio):int(len(abnormal_RFQ)*fault_end_ratio)]],
    # DTL module
    labels2[return_indicies(column_to_search_DTL, 'DTL3')[int(return_len(column_to_search_DTL,'DTL3')*fault_ratio):int(return_len(column_to_search_DTL, 'DTL3')*fault_end_ratio)]],
    labels2[return_indicies(column_to_search_DTL, 'DTL5')[int(return_len(column_to_search_DTL,'DTL5')*fault_ratio):int(return_len(column_to_search_DTL, 'DTL5')*fault_end_ratio)]],
    # CCL module
    labels3[return_indicies(column_to_search_CCL, 'CCL1')[int(return_len(column_to_search_CCL,'CCL1')*fault_ratio):int(return_len(column_to_search_CCL, 'CCL1')*fault_end_ratio)]],
    labels3[return_indicies(column_to_search_CCL, 'CCL2')[int(return_len(column_to_search_CCL,'CCL2')*fault_ratio):int(return_len(column_to_search_CCL, 'CCL2')*fault_end_ratio)]],
    labels3[return_indicies(column_to_search_CCL, 'CCL3')[int(return_len(column_to_search_CCL,'CCL3')*fault_ratio):int(return_len(column_to_search_CCL, 'CCL3')*fault_end_ratio)]],
    labels3[return_indicies(column_to_search_CCL, 'CCL4')[int(return_len(column_to_search_CCL,'CCL4')*fault_ratio):int(return_len(column_to_search_CCL, 'CCL4')*fault_end_ratio)]],
    # SCL module
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

# 사전학습 모델 초기화
pretrain_model = CVAE_rev(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim, condition_dim=condition_dim, dropout_prob=dropout).to(device)
pretrain_model.load_state_dict(torch.load('./model/0822/pretrained_all_waves_lr_schedueling.pth'))

# 사전학습 모델에서 디코더 모델 불러오기
state_dict = (torch.load('./model/0822/pretrained_all_waves_lr_schedueling.pth'))
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
        self.encoder.load_state_dict(torch.load('./model/0822/adaptation_all_waves_lr_scheduel_recon.pth'))
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
        self.encoder.load_state_dict(torch.load('./model/0822/adaptation_all_waves_lr_scheduel_recon.pth'))
        self.decoder.load_state_dict(decoder_state_dict)
        mu, logvar = self.encoder(x, c)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z, c)
        return x_recon, mu, logvar
    
test_normal = Valid_model_Normal(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim, condition_dim=condition_dim, dropout_prob=dropout).to(device)
test_fault = Valid_model_Fault(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim, condition_dim=condition_dim, dropout_prob=dropout).to(device)

# 손실 함수 및 옵티마이저
optimizer_normal = optim.Adam(test_normal.parameters(), lr=1e-3)
optimizer_fault = optim.Adam(test_fault.parameters(), lr=1e-3)
optimizer_pretrain = optim.Adam(pretrain_model.parameters(), lr=1e-3)

# Target encoder pretraining function
# to train discriminator
pretrain_model.eval()
test_normal.eval()
test_fault.eval()

with torch.no_grad():
    mse = []                # proposed model MSE
    mse_pretrain = []       # pretrain 모델 MSE
    
    # to load the dataset of normal signals
    data_test, labels = data_test.to(device), labels.to(device)
    
    # 기존 모델
    x_recon_pretrain, mu_normal_pretrain, logvar_normal_pretrain = pretrain_model(data_test, labels)

    # 제안한 모델 
    x_recon_2, mu_normal_proposed, logvar_normal_proposed = test_fault(data_test, labels)
    
    # 정상신호의 MSE 저장 
    mse_pretrain.extend((((data_test - x_recon_pretrain) ** 2)/4500).mean(dim=(1,2)).cpu().numpy()) # 기존모델을 이용해 복원한 정상신호의 MSE                                                                                                      
    mse.extend((((data_test - x_recon_2) ** 2)/4500).mean(dim=(1,2)).cpu().numpy()) # 제안된 모델을 이용해 복원한 정상신호의 MSE


with torch.no_grad():
    mse_error = []                          # proposed model error MSE          
    mse_error_pretrain = []                 # pretrain model error MSE
    
    # to load the dataset of abnormal signals
    data_anomaly, anomaly_labels = data_anomaly.to(device), anomaly_labels.to(device)
    
    # 기존 모델
    x_pretrain_error_recon, mu_abnormal_pretrain, logvar_abnormal_pretrain = pretrain_model(data_anomaly, anomaly_labels)
    
    # 제안한 모델
    x_error_recon_2, mu_abnormal_proposed, logvar_abnormal_proposed = test_fault(data_anomaly, anomaly_labels)

    # 비정상 신호의 MSE 저장
    mse_error_pretrain.extend((((data_anomaly - x_pretrain_error_recon) ** 2)/4500).mean(dim=(1,2)).cpu().numpy()) # 기존모델을 이용해 복원한 오류신호의 MSE
    mse_error.extend((((data_anomaly - x_error_recon_2) ** 2)/4500).mean(dim=(1,2)).cpu().numpy()) # 제안된 모델을 이용해 복원한 오류신호의 MSE
    

mse = np.array(mse)
mse_pretrain = np.array(mse_pretrain)
mse_error = np.array(mse_error)
mse_error_pretrain = np.array(mse_error_pretrain)

import pandas as pd

# 이상 탐지 정량화

######################################################################################################################################################################
######################################################################## 1. Mean and Standard ########################################################################
######################################################################################################################################################################

pretrain_normal_mean = np.mean(mse_pretrain)
pretrain_normal_std = np.std(mse_pretrain)

pretrain_fault_mean = np.mean(mse_error_pretrain)
pretrain_fault_std = np.std(mse_error_pretrain)

proposed_normal_mean = np.mean(mse)
proposed_normal_std = np.std(mse)

proposed_fault_mean = np.mean(mse_error)
proposed_fault_std = np.std(mse_error)

print("Pretrained Normal Loss - Mean: ", pretrain_normal_mean, " Std: ", pretrain_normal_std)
print("Pretrained Fault Loss - Mean: ", pretrain_fault_mean, " Std: ", pretrain_fault_std)
print("Proposed Normal Loss - Mean: ", proposed_normal_mean, " Std: ", proposed_normal_std)
print("Proposed Fault Loss - Mean: ", proposed_fault_mean, " Std: ", proposed_fault_std)

#####################################################################################################################################################################
######################################################################## 2. ROC and F1-score ########################################################################
#####################################################################################################################################################################

# 정상 데이터는 라벨 0, 비정상 데이터는 라벨 1로 가정
# true_labels = np.concatenate([np.zeros_like(mse_pretrain), np.ones_like(mse_error_pretrain)]) # pretrained model
true_labels = np.concatenate([np.zeros_like(mse), np.ones_like(mse_error)])   # proposed model

# MSE가 낮을수록 정상, 높을수록 비정상으로 예측한다고 가정
# 여기서는 MSE 값 자체를 점수로 사용합니다.
# scores = np.concatenate([mse_pretrain, mse_error_pretrain]) # pretrained model
scores = np.concatenate([mse, mse_error])                     # proposed model

# ROC 곡선 계산
fpr, tpr, thresholds = roc_curve(true_labels, scores)
 

########################## 2-1. Find the F1-score ##########################
# Youden's J 통계량 (J-Statistic) 사용
# Youden's J 통계량은 (TPR - FPR)의 최대값을 기준으로 최적의 threshold를 선택한다.
# 이 방법은 tpr과 fpr의 차이가 가장 큰 지점을 찾는다.
j_scores = tpr - fpr
optimal_idx = np.argmax(j_scores)
optimal_threshold = thresholds[optimal_idx]
# print( f'ROC Curve에서 정상신호와 비정산신호를 구분하는 최적의 값 : {optimal_threshold}')

predicted_labels = (scores >= optimal_threshold).astype(int)

# F1-score, precision, recall 계산
f1 = f1_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)

# 출력
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-score: {f1:.4f}')

# 혼동 행렬 출력
conf_matrix = confusion_matrix(true_labels, predicted_labels)
print('Confusion Matrix:')
print(conf_matrix)

# exit()

########################## 2-2. Plot the ROC Curve ##########################
roc_auc = auc(fpr, tpr)

# LaTeX 스타일 활성화
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# ROC 곡선 그리기
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('%s model ROC Curve'%args.fault)
plt.legend(loc="lower right")
plt.savefig('./figure/0822/roc/' +'%s'%args.fault + str(' Normal vs. Fault kde plot') + '.png', dpi=600)
