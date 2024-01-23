#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.cluster import DBSCAN
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import preprocessing
import scipy.io as scio
from time import time
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib as mpl
from matplotlib import pyplot as plt
#绘制混淆矩阵
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(conf_matrix):
    colors = ['#AADDFF', '#99CCFF', '#88BBEE', '#77AAEE']
    cmap = mpl.colors.ListedColormap(colors)
    plt.imshow(conf_matrix,cmap)
    indices = range(conf_matrix.shape[0])
    labels = [1,2,3]
    plt.xticks(indices, labels,fontsize = 25)
    plt.yticks(indices, labels,fontsize = 25)
    plt.rcParams['font.size'] = 16
    plt.colorbar()
    plt.xlabel('',fontdict={'family': 'Times New Roman', 'size': 25})
    plt.ylabel('',fontdict={'family': 'Times New Roman', 'size': 25})
    # 显示数据
    for first_index in range(conf_matrix.shape[0]):
        for second_index in range(conf_matrix.shape[1]):
            plt.text(first_index, second_index, conf_matrix[second_index, first_index],horizontalalignment='center', fontdict={'family' : 'Times New Roman', 'size'   : 25})
    plt.savefig('heatmap_confusion_matrix.jpg')
    plt.show()

def get_index1(lst=None, item=''):
    return [index for (index, value) in enumerate(lst) if value == item]

n = np.arange(1, 401)
sampleNo = 400
mu1 = 0
sigma1 = 0.1
mu2 = 0
sigma2 = 0.16
mu3 = 0
sigma3 = 0.18
mu_e1 = 0
sigma_e1 = 0.1414   #0.1414
mu_e2 = 0
sigma_e2 = 0.14
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
e21 = np.random.normal(mu_e2, sigma_e2, sampleNo)
e21 = e21[:, np.newaxis]

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
print(data_nomal.shape)

# z-score 标准化
scaler = preprocessing.StandardScaler().fit(data_nomal)  # 创建标准化转换器
X_training = preprocessing.scale(data_nomal)  # 标准化处理

# 训练数据聚类
X_training = X_training.T

t0 = time()
estimator = AgglomerativeClustering(linkage="ward", n_clusters=3)
estimator.fit(X_training)
label_pred = estimator.labels_

print("ward : %.2fs" % (time() - t0))

index1 = get_index1(label_pred, 0)
index2 = get_index1(label_pred, 1)
index3 = get_index1(label_pred, 2)
X_1 = X_training[index1, :].T
X_2 = X_training[index2, :].T
X_3 = X_training[index3, :].T
len1 = X_1.shape[1]
len2 = X_2.shape[1]
len3 = X_3.shape[1]
print(len1)
print(len2)
print(len3)
f = open("y_att.txt", "w")
f.writelines(str(index1))
f.writelines(str(index2))
f.writelines(str(index3))
f.close()

#添加噪声
e22 = np.random.normal(mu_e1, sigma_e2, sampleNo)
e22 = e22[:, np.newaxis]
e23 = np.random.normal(mu_e1, sigma_e2, sampleNo)
e23 = e23[:, np.newaxis]
e24 = np.random.normal(mu_e1, sigma_e2, sampleNo)
e24 = e24[:, np.newaxis]
e25 = np.random.normal(mu_e1, sigma_e2, sampleNo)
e25 = e25[:, np.newaxis]
e26 = np.random.normal(mu_e2, sigma_e2, sampleNo)
e26 = e26[:, np.newaxis]
E_1_3 = np.concatenate((e22, e23,e22, e23,e22,e22), axis=1)


X_1 = torch.from_numpy(X_1)
X_2 = torch.from_numpy(X_2)
X_3 = torch.from_numpy(X_3)
X_4 = torch.from_numpy(E_1_3)

Q_1 = F.softmax(X_1, dim=-1)
Q_2 = F.softmax(X_2, dim=-1)
Q_3 = F.softmax(X_3, dim=-1)
Q_4 = F.softmax(X_4, dim=-1)

X_1_2 = torch.cat((X_1, X_2), dim=1)
X_1_3 = torch.cat((X_1, X_3), dim=1)
X_1_4 = torch.cat((X_1, X_4), dim=1)
X_2_3 = torch.cat((X_2, X_3), dim=1)
X_2_4 = torch.cat((X_2, X_4), dim=1)
X_3_4 = torch.cat((X_3, X_4), dim=1)

Q_1_2 = F.softmax(X_1_2, dim=-1)
Q_1_3 = F.softmax(X_1_3, dim=-1)
Q_1_4 = F.softmax(X_1_4, dim=-1)
Q_2_3 = F.softmax(X_2_3, dim=-1)
Q_2_4 = F.softmax(X_2_4, dim=-1)
Q_3_4 = F.softmax(X_3_4, dim=-1)

H_1 = torch.mean(-Q_1 * torch.log(Q_1), dim=1) * X_1.shape[1]
H_2 = torch.mean(-Q_2 * torch.log(Q_2), dim=1) * X_2.shape[1]
H_3 = torch.mean(-Q_3 * torch.log(Q_3), dim=1) * X_3.shape[1]
H_4 = torch.mean(-Q_4 * torch.log(Q_4), dim=1) * X_4.shape[1]

H_1_2 = torch.mean(-Q_1_2 * torch.log(Q_1_2), dim=1) * (X_1.shape[1]+X_2.shape[1])
H_1_3 = torch.mean(-Q_1_3 * torch.log(Q_1_3), dim=1) * (X_1.shape[1]+X_3.shape[1])
H_1_4 = torch.mean(-Q_1_4 * torch.log(Q_1_4), dim=1) * (X_1.shape[1]+X_4.shape[1])
H_2_3 = torch.mean(-Q_2_3 * torch.log(Q_2_3), dim=1) * (X_2.shape[1]+X_3.shape[1])
H_2_4 = torch.mean(-Q_2_4 * torch.log(Q_2_4), dim=1) * (X_2.shape[1]+X_4.shape[1])
H_3_4 = torch.mean(-Q_3_4 * torch.log(Q_3_4), dim=1) * (X_3.shape[1]+X_4.shape[1])

M1_2 = H_1 + H_2 - H_1_2
M1_3 = H_1 + H_3 - H_1_3
M1_4 = H_1 + H_4 - H_1_4
M2_3 = H_2 + H_3 - H_2_3
M2_4 = H_2 + H_4 - H_2_4
M3_4 = H_3 + H_4 - H_3_4

M1_2 = torch.mean(M1_2).numpy()
M1_3 = torch.mean(M1_3).numpy()
M1_4 = torch.mean(M1_4).numpy()
M2_3 = torch.mean(M2_3).numpy()
M2_4 = torch.mean(M2_4).numpy()
M3_4 = torch.mean(M3_4).numpy()

M1_2 = np.around(M1_2,decimals=2)
M1_3 = np.around(M1_3,decimals=2)
M1_4 = np.around(M1_3,decimals=2)
M2_3 = np.around(M2_3,decimals=2)
M2_4 = np.around(M2_4,decimals=2)
M3_4 = np.around(M3_4,decimals=2)
print(M1_2)
print(M1_3)
print(M2_3)

block_1_1 = np.around(torch.mean(H_1).numpy(),decimals=2)
block_2_2 = np.around(torch.mean(H_2).numpy(),decimals=2)
block_3_3 = np.around(torch.mean(H_3).numpy(),decimals=2)
block_4_4 = np.around(torch.mean(H_4).numpy(),decimals=2)

block_1_2 = M1_2
block_1_3 = M1_3
block_1_4 = M1_4

block_2_3 = M2_3
block_2_4 = M2_4
block_3_4 = M3_4

list = [block_1_1,block_1_2,block_1_3,block_2_2,block_2_3,block_3_3]
matrix = np.zeros((3,3))
for i in range (matrix.shape[0]):
    for j in range(i,matrix.shape[1]):
        if i == 0:
            matrix[i][j] = round(list[i+j],2)
        if i ==1:
            matrix[i][j] = round(list[i + j +1],2)
        if i ==2:
            matrix[i][j] = round(list[i + j +1],2)
        if i ==3:
            matrix[i][j] = round(list[i + j +3],2)
matrix += matrix.T - np.diag(matrix.diagonal())
plot_confusion_matrix(matrix)