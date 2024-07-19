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
from model import *

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

# 원-핫 인코딩
labels_concat = np.concatenate((labels1[normal_indices_RFQ[:524]], labels2[normal_indices_DTL[:524]], labels3[normal_indices_CCL[:524]], labels4[normal_indices_SCL[:524]]), axis=0)
labels_one_hot = np.eye(len(system_indices))[labels_concat]

# Min-Max 스케일러 인스턴스 초기화
# scaler = MinMaxScaler()
scaler_IGBT = MinMaxScaler()
scaler_FLUX = MinMaxScaler()
scaler_CAP = MinMaxScaler()
scaler_MOD = MinMaxScaler()

Xnormal_concat = np.concatenate( (Xnormal_RFQ[:160,:,:], Xnormal_DTL[:160,:,:], Xnormal_CCL[:160,:,:], Xnormal_SCL[:160,:,:]), axis=0 )
Xnormal_concat = np.array(Xnormal_concat)

# Target Encoder가 학습할 데이터 샘플 160개
column_to_search_RFQ = Yanomaly_RFQ[:, 2].astype(str)
column_to_search_DTL = Yanomaly_DTL[:, 2].astype(str)
column_to_search_CCL = Yanomaly_CCL[:, 2].astype(str)
column_to_search_SCL = Yanomaly_SCL[:, 2].astype(str)

fault_RFQ = np.where(np.char.find(column_to_search_RFQ, f'{args.fault}') != -1)[0]
len_fault_RFQ = len(fault_RFQ) 

fault_DTL = np.where(np.char.find(column_to_search_DTL, f'{args.fault}') != -1)[0]
len_fault_DTL = len(fault_DTL)

fault_CCL = np.where(np.char.find(column_to_search_CCL, f'{args.fault}') != -1)[0]
len_fault_CCL = len(fault_CCL)

fault_SCL = np.where(np.char.find(column_to_search_SCL, f'{args.fault}') != -1)[0]
len_fault_SCL = len(fault_SCL)

# 테스트 : normal / abnormal 설정
sample = 160
data_test = np.concatenate( (Xnormal_RFQ[524:524+sample,:,:], Xnormal_DTL[524:524+sample,:,:], Xnormal_CCL[524:524+sample,:,:], Xnormal_SCL[524:524+sample,:,:]), axis=0 )
data_anomaly = np.concatenate(
    (Xanomaly_RFQ[fault_RFQ,:,:],
     Xanomaly_DTL[fault_DTL,:,:],
     Xanomaly_CCL[fault_CCL,:,:],
     Xanomaly_SCL[fault_SCL,:,:]), axis=0)

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
tmp_labels = np.concatenate((labels_one_hot[:sample,:], labels_one_hot[524:524+sample,:],labels_one_hot[524*2:524*2+sample,:],labels_one_hot[524*3:524*3+sample,:]))
labels = torch.tensor(tmp_labels, dtype=torch.float32)

# GPU 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 및 학습파라미터 설정
input_dim = 14
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
state_dict = (torch.load('./model/CVAE_pretrained.pth'))
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
        self.encoder.load_state_dict(torch.load('./model/target_encoder_extended_normal.pth'))
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
        self.encoder.load_state_dict(torch.load('./model/target_encoder_extended_error.pth'))
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
        ).to(device).to(device)

test_fault = Valid_model_Fault(
        input_dim=input_dim, 
        # seq_len=input_dim, 
        latent_dim=latent_dim,
        hidden_dim=hidden_dim, 
        condition_dim=condition_dim,
        dropout_prob=dropout
        ).to(device).to(device)

# 손실 함수 및 옵티마이저
optimizer_normal = optim.Adam(test_normal.parameters(), lr=1e-5)
optimizer_fault = optim.Adam(test_fault.parameters(), lr=1e-5)


# Target encoder pretraining function
if __name__ =='__main__':
    # to train discriminator
    test_normal.eval()
    with torch.no_grad():
        mse = []
        mse_error = []
        labels_list = []
        data_test, labels = data_test.to(device), labels.to(device)
        x_recon_1, _, _ = test_normal(data_test, labels)
        x_recon_2, _, _ = test_fault(data_test, labels)
        x_recon = (x_recon_1 + x_recon_2) / 2
        mse.extend(((data_test - x_recon) ** 2).mean(dim=(1,2)).cpu().numpy())

    anomaly_labels = np.concatenate((labels_one_hot[:len_fault_RFQ,:],labels_one_hot[524*1:524*1+len_fault_DTL,:],labels_one_hot[524*2:524*2+len_fault_CCL,:],labels_one_hot[524*3:524*3+len_fault_SCL,:]), axis=0)
    anomaly_labels = torch.tensor(anomaly_labels, dtype=torch.float32)
    test_fault.eval()
    with torch.no_grad():
        mse_error = []
        labels_list = []
        data_anomaly, anomaly_labels = data_anomaly.to(device), anomaly_labels.to(device)
        x_error_recon_1, _, _ = test_normal(data_anomaly, anomaly_labels)
        x_error_recon_2, _, _ = test_fault(data_anomaly, anomaly_labels)
        x_error_recon = (x_error_recon_1 + x_error_recon_2) / 2
        mse_error.extend(((data_anomaly - x_error_recon) ** 2).mean(dim=(1,2)).cpu().numpy())

    mse = np.array(mse)
    mse_error = np.array(mse_error)
    labels_list = np.array(labels_list)

    # LaTeX 스타일 활성화
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    
    # 이상 탐지 시각화
    for i in range(0,1):
        plt.figure(figsize=(12, 6))
        plt.hist(mse, bins=100, alpha=0.5, label='Normal data',color='gray')
        plt.hist(mse_error, bins=100, alpha=0.5, label='%s'%args.fault+' Fault',color='red')
        sns.kdeplot((mse), fill=True,label='Normal data',color='gray',cut=0)
        sns.kdeplot((mse_error), fill=True,label='%s'%args.fault+' Fault',color='red',cut=0)

        # sci 논문 양식
        # plt.grid(True, which='both', linestyle='--', linewidth=0.5) # grid 설정
        plt.legend(fontsize=12)
        plt.xlabel('MSE', fontsize=14, fontweight='bold')
        plt.ylabel('Density', fontsize=14, fontweight='bold')
        plt.title(f'Normal vs. %s'%args.fault+ ' Fault')

        plt.xscale('log')
        plt.xlim(1e-3, 1)
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'$10^{{{int(np.log10(x))}}}$'))

        if f'{args.fault}' == 'DV/DT' :
            args.fault = 'DV_DT'
        plt.savefig('./figure/0719/' + str('Normal vs. %s'%args.fault+' Fault') + '.png', dpi=600)
