#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import preprocessing
import scipy.io as scio
from torch.utils.data.sampler import SubsetRandomSampler
from pytorchtools import EarlyStopping
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import datetime
import os


def load_te_detection_data(test=5):
    n_th = 200
    alpha = 0.95
    perf_T2 = np.zeros([2, 15])
    perf_Q = np.zeros([2, 15])  # 记录各个尺度的检测性能

    n = np.arange(1, 401)
    sampleNo = 400
    mu1 = 0
    sigma1 = 0.1
    mu2 = 0
    sigma2 = 0.16
    mu3 = 0
    sigma3 = 0.18
    mu_e1 = 0
    sigma_e1 = 0.1414
    np.random.seed(0)
    s1 = np.random.normal(mu1, sigma1, sampleNo)
    s1 = s1[:, np.newaxis]
    s2 = np.random.normal(mu1, sigma1, sampleNo)
    s2 = s2[:, np.newaxis]
    s3 = np.random.normal(mu1, sigma1, sampleNo)
    s3 = s3[:, np.newaxis]
    s4 = np.random.normal(mu2, sigma2, sampleNo)
    s4 = s4[:, np.newaxis]
    s5 = np.random.normal(mu2, sigma2, sampleNo)
    s5 = s5[:, np.newaxis]
    s6 = np.random.normal(mu2, sigma2, sampleNo)
    s6 = s6[:, np.newaxis]
    s7 = np.random.normal(mu3, sigma3, sampleNo)
    s7 = s7[:, np.newaxis]
    s8 = np.random.normal(mu3, sigma3, sampleNo)
    s8 = s8[:, np.newaxis]
    s9 = np.random.normal(mu3, sigma3, sampleNo)
    s9 = s9[:, np.newaxis]

    e1 = np.random.normal(mu_e1, sigma_e1, sampleNo)
    e1 = e1[:, np.newaxis]
    e2 = np.random.normal(mu_e1, sigma_e1, sampleNo)
    e2 = e2[:, np.newaxis]
    e3 = np.random.normal(mu_e1, sigma_e1, sampleNo)
    e3 = e3[:, np.newaxis]
    e4 = np.random.normal(mu_e1, sigma_e1, sampleNo)
    e4 = e4[:, np.newaxis]
    e5 = np.random.normal(mu_e1, sigma_e1, sampleNo)
    e5 = e5[:, np.newaxis]
    e6 = np.random.normal(mu_e1, sigma_e1, sampleNo)
    e6 = e6[:, np.newaxis]
    e7 = np.random.normal(mu_e1, sigma_e1, sampleNo)
    e7 = e7[:, np.newaxis]
    e8 = np.random.normal(mu_e1, sigma_e1, sampleNo)
    e8 = e8[:, np.newaxis]
    e9 = np.random.normal(mu_e1, sigma_e1, sampleNo)
    e9 = e9[:, np.newaxis]
    e10 = np.random.normal(mu_e1, sigma_e1, sampleNo)
    e10 = e10[:, np.newaxis]
    e11 = np.random.normal(mu_e1, sigma_e1, sampleNo)
    e11 = e11[:, np.newaxis]
    e12 = np.random.normal(mu_e1, sigma_e1, sampleNo)
    e12 = e12[:, np.newaxis]
    e13 = np.random.normal(mu_e1, sigma_e1, sampleNo)
    e13 = e13[:, np.newaxis]
    e14 = np.random.normal(mu_e1, sigma_e1, sampleNo)
    e14 = e14[:, np.newaxis]
    e15 = np.random.normal(mu_e1, sigma_e1, sampleNo)
    e15 = e15[:, np.newaxis]
    e16 = np.random.normal(mu_e1, sigma_e1, sampleNo)
    e16 = e16[:, np.newaxis]
    e17 = np.random.normal(mu_e1, sigma_e1, sampleNo)
    e17 = e17[:, np.newaxis]
    e18 = np.random.normal(mu_e1, sigma_e1, sampleNo)
    e18 = e18[:, np.newaxis]
    e19 = np.random.normal(mu_e1, sigma_e1, sampleNo)
    e19 = e19[:, np.newaxis]
    e20 = np.random.normal(mu_e1, sigma_e1, sampleNo)
    e20 = e20[:, np.newaxis]
    # print(s6.shape)
    S1_3 = np.concatenate((s1, s2, s3), axis=1)
    S4_6 = np.concatenate((s4, s5, s6), axis=1)
    S7_9 = np.concatenate((s7, s8, s9), axis=1)
    E1_5 = np.concatenate((e1, e2, e3, e4, e5), axis=1)
    E6_10 = np.concatenate((e6, e7, e8, e9, e10), axis=1)
    E11_15 = np.concatenate((e11, e12, e13, e14, e15), axis=1)

    matrix1 = np.array(
        [[1.57, 1.37, 1.80], [1.73, 1.05, 1.70], [1.82, 1.40, 1.60], [1.65, 1.20, 1.50], [1.47, 1.24, 1.60]])
    matrix2 = np.array(
        [[1.67, 1.47, 1.70], [1.63, 1.15, 1.80], [1.72, 1.30, 1.70], [1.55, 1.30, 1.60], [1.45, 1.38, 1.80]])
    matrix3 = np.array(
        [[1.58, 1.92, 1.47], [1.53, 1.20, 1.26], [1.45, 1.53, 1.79], [1.86, 1.76, 1.89], [1.77, 1.73, 1.53]])

    Matraix_X1_5 = np.matmul(S1_3, matrix1.T) + 0.01 * E1_5
    Matraix_X6_10 = np.matmul(S4_6, matrix2.T) + 0.01 * E6_10
    Matraix_X11_15 = np.matmul(S7_9, matrix3.T) + 0.01 * E11_15
    # 耦合数据
    x1 = Matraix_X1_5[:, 0].reshape(-1, 1)
    x2 = Matraix_X1_5[:, 1].reshape(-1, 1)
    x3 = Matraix_X1_5[:, 2].reshape(-1, 1)
    x6 = Matraix_X6_10[:, 0].reshape(-1, 1)
    x7 = Matraix_X6_10[:, 1].reshape(-1, 1)
    x11 = Matraix_X11_15[:, 0].reshape(-1, 1)
    x12 = Matraix_X11_15[:, 1].reshape(-1, 1)
    x13 = Matraix_X11_15[:, 2].reshape(-1, 1)

    X16 = x1 ** 2 + x2 ** 2 + x3 ** 2 + e16 * 0.01
    X17 = x1 ** 3 + 2 * x6 ** 2 + e17 * 0.01
    X18 = x6 + x11 + 3 * x13 + e18 * 0.01
    X19 = x2 ** 2 + x11 ** 2 + e19 * 0.01
    X20 = x1 ** 2 + x7 ** 2 + x12 ** 2 + e20 * 0.01
    Matraix_X16_20 = np.concatenate((X16, X17, X18, X19, X20), axis=1)
    data_nomal = np.concatenate((Matraix_X1_5, Matraix_X6_10, Matraix_X11_15, Matraix_X16_20), axis=1)

    # z-score 标准化
    scaler = preprocessing.StandardScaler().fit(data_nomal)  # 创建标准化转换器
    X_training = preprocessing.scale(data_nomal)  # 标准化处理
    # X_training = np.unsqueeze(X_training, axis=2)

    return X_training


def creat_torch_datasets(batch_size):
    valid_size = 0.2

    train_data = load_te_detection_data()

    train_data = torch.FloatTensor(train_data)

    train_data = TensorDataset(train_data, train_data)

    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(num_train * valid_size))

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=batch_size,
                                               sampler=train_sampler,
                                               num_workers=0)

    valid_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=batch_size,
                                               sampler=valid_sampler,
                                               num_workers=0)

    return train_loader, valid_loader


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(20, 40),
            nn.Tanh(),
            nn.Linear(40, 24),
            nn.Tanh(),
            nn.Linear(24, 4),
            nn.Tanh())

        self.decoder = nn.Sequential(
            nn.Linear(4, 24),
            nn.Tanh(),
            nn.Linear(24, 40),
            nn.Tanh(),
            nn.Linear(40, 20))

    def forward(self, x):
        x_encoded = self.encoder(x)
        x_recon = self.decoder(x_encoded)
        return x_encoded, x_recon


autoencoder = Net()
print(autoencoder)
print('@' * 70)

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=20)

creterion = nn.MSELoss()


def train_model(model, n_epochs, batch_size, patience):
    train_losses = []

    valid_losses = []

    avg_train_loss = []

    avg_valid_loss = []

    lr_his = []

    early_stopping_path = 'SAE_checkpoint.pt'
    early_stopping = EarlyStopping(patience, verbose=True, path=early_stopping_path)
    log_dir = os.path.join('logs', datetime.datetime.now().strftime('%Y%m%d %H%M%S'))

    for epoch in range(1, n_epochs + 1):

        model.train()
        for batch, (data, _) in enumerate(train_loader):

            optimizer.zero_grad()
            feature, output = model(data)

            loss = creterion(output, data)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())


        model.eval()
        for batch, (data, _) in enumerate(valid_loader):
            feature, output = model(data)

            loss = creterion(output, data)
            valid_losses.append(loss.item())

        lr_schedule.step(loss)

        lr = optimizer.param_groups[0]['lr']

        lr_his.append(lr)

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_loss.append(train_loss)
        avg_valid_loss.append(valid_loss)

        epoch_len = len(str(n_epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}]' +
                     f'train_loss:{train_loss:.5f}' +
                     f'valid_loss:{valid_loss:.5f}')

        print(print_msg)

        train_losses = []
        valid_losses = []

        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print('early_stop')
            break

    model.load_state_dict(torch.load(early_stopping_path))

    return model, avg_train_loss, avg_valid_loss, lr_his


batch_size = 10
n_epochs = 2000

train_loader, valid_loader = creat_torch_datasets(batch_size)

patience = 40

model, train_loss, valid_loss, lr_his = train_model(autoencoder, n_epochs, batch_size, patience)

# visualizing the loss and the early stopping checkpoint
# visualize the loss as the network trained
fig = plt.figure(figsize=(10, 8), dpi=150)
plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
plt.plot(range(1, len(valid_loss) + 1), valid_loss, label='Validation Loss')

# find positing of lowest validation loss
minposs = valid_loss.index(min(valid_loss)) + 1
plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

plt.xlabel('epochs')
plt.ylabel('loss')
plt.ylim(0, 0.5)
plt.xlim(0, len(train_loss) + 1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
fig.savefig('SAE_loss_ploy.png', bbox_inches='tight')

plt.plot(lr_his)
