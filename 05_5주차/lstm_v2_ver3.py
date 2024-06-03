import numpy as np 
import matplotlib.pyplot as plt
import os
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

min_max_scaler = MinMaxScaler()

# User parameters
features=['A+IGBT-I', 'A+*IGBT-I', 'B+IGBT-I', 'B+*IGBT-I', 'C+IGBT-I', 
          'C+*IGBT-I', 'A-FLUX', 'B-FLUX', 'C-FLUX', 'MOD-V', 'MOD-I', 'CB-I', 
          'CB-V', 'DV/DT']
feature_index=6  # A-flux waveform
system='RFQ'     # pick a system

# Load waveform (X) and labels (Y) datasets
X= np.load('./hvcm/data/hvcm/%s.npy' % system)
Y=np.load('./hvcm/data/hvcm/%s_labels.npy' % system, allow_pickle=True)
time=np.arange(X.shape[1]) * 400e-9

# Split X/Y arrays into fault and normal data
fault_indices, normal_indices = np.where(Y[:,1] == 'Fault')[0], np.where(Y[:,1] == 'Run')[0]
Xnormal, Xanomaly = X[normal_indices,:,:], X[fault_indices,:,:]
Ynormal, Yanomaly = Y[normal_indices,:], Y[fault_indices,:]

class lstm_encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3):
        super(lstm_encoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, x_input):
        lstm_out, self.hidden = self.lstm(x_input)
        return lstm_out, self.hidden

class lstm_decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3):
        super(lstm_decoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x_input, encoder_hidden_states):
        lstm_out, self.hidden = self.lstm(x_input.unsqueeze(-1), encoder_hidden_states)
        output = self.linear(lstm_out)
        return output, self.hidden

class lstm_encoder_decoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(lstm_encoder_decoder, self).__init__()
        self.encoder = lstm_encoder(input_size, hidden_size)
        self.decoder = lstm_decoder(input_size, hidden_size)

    def forward(self, inputs, targets, target_len, teacher_forcing_ratio):
        batch_size = inputs.shape[0]
        input_size = inputs.shape[2]
        outputs = torch.zeros(batch_size, target_len, input_size).to(inputs.device)

        _, hidden = self.encoder(inputs)
        decoder_input = inputs[:, -1, :]
        
        for t in range(target_len): 
            out, hidden = self.decoder(decoder_input, hidden)
            out =  out.squeeze(1)
            
            if random.random() < teacher_forcing_ratio:
                decoder_input = targets[:, t, :]
            else:
                decoder_input = out
            outputs[:, t, :] = out
        return outputs

    def predict(self, inputs, target_len):
        inputs = inputs.unsqueeze(0)
        self.eval()
        batch_size = inputs.shape[0]
        input_size = inputs.shape[2]
        outputs = torch.zeros(batch_size, target_len, input_size).to(inputs.device)
        _, hidden = self.encoder(inputs)
        decoder_input = inputs[:, -1, :]
        for t in range(target_len): 
            out, hidden = self.decoder(decoder_input, hidden)
            out =  out.squeeze(1)
            decoder_input = out
            outputs[:, t, :] = out
        return outputs.detach().cpu().numpy()[0, :, 0]

class windowDataset(Dataset):
    def __init__(self, y, input_window=100, output_window=50, stride=5):
        L = y.shape[0]
        num_samples = (L - input_window - output_window) // stride + 1

        X = np.zeros([input_window, num_samples])
        Y = np.zeros([output_window, num_samples])

        for i in np.arange(num_samples):
            start_x = stride * i
            end_x = start_x + input_window
            X[:, i] = y[start_x:end_x]

            start_y = stride * i + input_window
            end_y = start_y + output_window
            Y[:, i] = y[start_y:end_y]

        X = X.reshape(X.shape[0], X.shape[1], 1).transpose((1, 0, 2))
        Y = Y.reshape(Y.shape[0], Y.shape[1], 1).transpose((1, 0, 2))
        self.x = X
        self.y = Y
        self.len = len(X)
        
    def __getitem__(self, i):
        return self.x[i], self.y[i]
    
    def __len__(self):
        return self.len

a_flux_fault_indices = np.where(Yanomaly[:, 2] == 'A FLUX Low Fault')[0]
a_flux_anomaly = Xanomaly[a_flux_fault_indices, :, :]
a_flux_anomaly_append = min_max_scaler.fit_transform(a_flux_anomaly[0, :, feature_index].reshape(-1, 1))

for i in range(1, len(a_flux_fault_indices)):
    tmp = min_max_scaler.fit_transform(a_flux_anomaly[i, :, feature_index].reshape(-1, 1))
    a_flux_anomaly_append = np.append(a_flux_anomaly_append, tmp)

train = a_flux_anomaly_append.flatten()

iw = 100  # 입력 시퀀스 길이 조정
ow = 50  # 출력 시퀀스 길이 조정

train_dataset = windowDataset(train, input_window=iw, output_window=ow, stride=1)
train_loader = DataLoader(train_dataset, batch_size=64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = lstm_encoder_decoder(input_size=1, hidden_size=128).to(device)

learning_rate = 0.0001  # 학습률 조정
epoch = 100  # 에포크 설정
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

predict = model.predict(torch.tensor(train_dataset[0][0]).to(device).float(), target_len=ow)
real = train_dataset[0][1]

predict_time = time[-int(ow):]
real_time = time[-int(ow):]

plt.figure()
plt.plot(real_time, predict)
plt.plot(real_time, real)
plt.legend(["Predicted", "Real"])
plt.xlabel('Time (s)')
plt.ylabel(features[feature_index])
plt.savefig("./figure/before_gpt_ver2", dpi=600)

from tqdm import tqdm

model.train()
with tqdm(range(epoch)) as tr:
    for epoch in tr:
        total_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            x = x.to(device).float()
            y = y.to(device).float()
            output = model(x, y, ow, 0.6).to(device)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.cpu().item()
        tr.set_postfix(loss="{0:.5f}".format(total_loss / len(train_loader)))

torch.save(model.state_dict(), './model/lstm_ver3.pth')

model_load = lstm_encoder_decoder(input_size=1, hidden_size=128).to(device)
model_load.load_state_dict(torch.load('./model/lstm_ver3.pth'))

predict = model_load.predict(torch.tensor(train[-int(ow):]).reshape(-1, 1).to(device).float(), target_len=ow)
real = train[-int(ow):]

predict = min_max_scaler.inverse_transform(predict.reshape(-1, 1))
real = min_max_scaler.inverse_transform(real.reshape(-1, 1))

plt.figure()
plt.plot(real_time, real[-int(ow):])
plt.plot(real_time, predict)
plt.legend(["Real", "Predicted"])
plt.xlabel('Time (s)')
plt.ylabel(features[feature_index])
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.savefig("./figure/after_gpt_ver2", dpi=600)
