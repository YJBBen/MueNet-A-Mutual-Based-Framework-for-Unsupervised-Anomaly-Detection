# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 09:22:06 2018

@author: Administrator
"""
import scipy
from scipy import stats
import scipy.io as scio
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing


def cal_threshold(x, alpha):
    kernel = stats.gaussian_kde(x)
    step = np.linspace(min(x), max(x), 10000)
    pdf = kernel(step)
    for i in range(len(step)):
        if sum(pdf[0:(i + 1)]) / sum(pdf) > alpha:
            break
    return step[i + 1]


def PCA(NewXtran):
    size = np.shape(NewXtran)
    U, S, V = np.linalg.svd(NewXtran / math.sqrt(size[0] - 1))
    eigVals = S ** 2  # 方差
    arraySum = sum(eigVals)
    tmpSum = 0
    n = 0
    for i in eigVals:
        tmpSum += i
        if tmpSum / arraySum < 0.85:
            n += 1
    lamda = eigVals[0:n + 1]
    n_eigVect = V[:, 0:n + 1]
    f_eigVect = V[:, n + 1:size[1]]
    return n, lamda, n_eigVect, f_eigVect  # 保留主元的个数，n_eigVect主元，f_eigVect残差


# 离线训练
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

matrix1 = np.array([[1.57, 1.37, 1.80], [1.73, 1.05, 1.70], [1.82, 1.40, 1.60], [1.65, 1.20, 1.50], [1.47, 1.24, 1.60]])
matrix2 = np.array([[1.67, 1.47, 1.70], [1.63, 1.15, 1.80], [1.72, 1.30, 1.70], [1.55, 1.30, 1.60], [1.45, 1.38, 1.80]])
matrix3 = np.array([[1.58, 1.92, 1.47], [1.53, 1.20, 1.26], [1.45, 1.53, 1.79], [1.86, 1.76, 1.89], [1.77, 1.73, 1.53]])

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
scaler = preprocessing.StandardScaler().fit(data_nomal)  # 创建标准化转换器
X_training = preprocessing.scale(data_nomal)  # 标准化处理

# 构造异常数据fault1
# print(s1[200:,:].shape)
s1_fault = np.concatenate((s1[:200, :], s1[200:, :] + 0.1), axis=0)
S1_3_fault = np.concatenate((s1_fault, s2, s3), axis=1)
Matraix_X1_5_fault = np.matmul(S1_3_fault, matrix1.T) + 0.01 * E1_5
x1 = Matraix_X1_5_fault[:, 0].reshape(-1, 1)
x2 = Matraix_X1_5_fault[:, 1].reshape(-1, 1)
x3 = Matraix_X1_5_fault[:, 2].reshape(-1, 1)
X16 = x1 ** 2 + x2 ** 2 + x3 ** 2 + e16 * 0.01
X17 = x1 ** 3 + 2 * x6 ** 2 + e17 * 0.01
X18 = x6 + x11 + 3 * x13 + e18 * 0.01
X19 = x2 ** 2 + x11 ** 2 + e19 * 0.01
X20 = x1 ** 2 + x7 ** 2 + x12 ** 2 + e20 * 0.01
Matraix_X16_20_fault1 = np.concatenate((X16, X17, X18, X19, X20), axis=1)
fault1 = np.concatenate((Matraix_X1_5_fault, Matraix_X6_10, Matraix_X11_15, Matraix_X16_20_fault1), axis=1)
# 构造异常数据fault2
ramp = np.linspace(0, 0.1, 200).reshape(-1, 1)
# # # print(ramp.shape)
s5_fault = np.concatenate((s5[:200, :], s5[200:, :] + ramp), axis=0)
S4_6_fault = np.concatenate((s4, s5_fault, s6), axis=1)
Matraix_X6_10_fault = np.matmul(S4_6_fault, matrix2.T) + 0.01 * E6_10
x6 = Matraix_X6_10_fault[:, 0].reshape(-1, 1)
x7 = Matraix_X6_10_fault[:, 1].reshape(-1, 1)
X16 = x1 ** 2 + x2 ** 2 + x3 ** 2 + e16 * 0.01
X17 = x1 ** 3 + 2 * x6 ** 2 + e17 * 0.01
X18 = x6 + x11 + 3 * x13 + e18 * 0.01
X19 = x2 ** 2 + x11 ** 2 + e19 * 0.01
X20 = x1 ** 2 + x7 ** 2 + x12 ** 2 + e20 * 0.01
Matraix_X16_20_fault2 = np.concatenate((X16, X17, X18, X19, X20), axis=1)

fault2 = np.concatenate((Matraix_X1_5, Matraix_X6_10_fault, Matraix_X11_15, Matraix_X16_20_fault2), axis=1)
print(fault2.shape)
data_testing = scaler.transform(fault2)  # 测试数据标准化
X_test = data_testing
n_test = X_test.shape[0]
# print(Xtran)
size = np.shape(data_nomal)  # 读取数据尺寸
mean = np.mean(data_nomal, axis=0)  # 平均值
std = np.std(data_nomal, axis=0)  # 方差
NewXtran = np.zeros((size[0], size[1]))  # 数据标准化
for i in range(size[1]):
    NewXtran[:, i] = (data_nomal[:, i] - mean[i]) / std[i]
n, lamda, n_eigVect, f_eigVect = PCA(NewXtran)  # PCA处理
print(n)
print(lamda.shape)
print(n_eigVect.shape)
print(f_eigVect.shape)

# 在线检测
Xte = X_test
Sz = np.shape(Xte)  # 读取测试数据尺寸
Xtest = np.zeros((Sz[0], Sz[1]))  # 测试数据标准化
for i in range(Sz[1]):
    Xtest[:, i] = (Xte[:, i] - mean[i]) / std[i]

# 计算T2统计量
T = np.zeros((Sz[0], 1))
D = np.diag(lamda)
for l in range(Sz[0]):
    T[l, 0] = np.dot(np.dot(np.dot(np.dot(Xtest[l, :], n_eigVect), np.linalg.inv(D)), n_eigVect.T), Xtest.T[:, l])  # @3
    # print(T2UCL-T[l, 0])
# 计算SPE统计量
Q = np.zeros((Sz[0], 1))
for s in range(Sz[0]):
    r = np.dot(Xtest[s, :], (np.identity(Sz[1]) - np.dot(f_eigVect, f_eigVect.T)))
    Q[s, 0] = np.dot(r, r.T)  # @4

print("---------")
T = T.squeeze(1)
Q = Q.squeeze(1)

T2UCL = cal_threshold(T[0:n_th], alpha)
QUCL = cal_threshold(Q[0:n_th], alpha)
print(T2UCL)
print(QUCL)
# 绘图

t = np.arange(0, 400)
T2A = np.ones((400, 1)) * T2UCL
QA = np.ones((400, 1)) * QUCL

plt.figure(figsize=(7, 5))
plt.subplot(211)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                    wspace=None, hspace=0.4)

# scipy.io.savemat("./data_mat/PCA/fault2_T2_data.mat", {"PCAdata2T": T})
# scipy.io.savemat("./data_mat/PCA/fault2_T2_limit.mat", {"PCAlimit2T": T2A})
plt.plot(T, 'b', lw=1.5, label='mornitoring index')  # 参数控制颜色和字体的粗细
plt.plot(T2A, 'r', label='control limit')
plt.grid(True)
plt.legend(loc=0, fontsize=14)
plt.title('Mornitoring performance of PCA', fontsize=20)
plt.xlabel('Sample number', fontsize=14)
plt.ylabel('T²', fontsize=14)
# 画图Q
# plt.figure(figsize=(7,4))
# scipy.io.savemat("./data_mat/PCA/fault2_Q_data.mat", {"PCAdata2Q": Q})
# scipy.io.savemat("./data_mat/PCA/fault2_Q_limit.mat", {"PCAlimit2Q": QA})
plt.subplot(212)
plt.plot(Q, 'b', lw=1.5, label='mornitoring index')  # 参数控制颜色和字体的粗细
plt.plot(QA, 'r', label='control limit')
plt.grid(True)
plt.legend(loc=0, fontsize=14)
plt.title('Mornitoring performance of PCA', fontsize=20)
plt.xlabel('Sample number', fontsize=14)
plt.ylabel('SPE', fontsize=14)
plt.show()

# 计算误报率和检测率
# 误报率
mn0 = 0
for i in range(0, 200):
    if T[i] > T2UCL:
        mn0 = mn0 + 1

mn0 = mn0 / 200
print("The false alarm rate of PCA T2 is: " + str(mn0))

# 检测率
mn1 = 0
for i in range(200, 400):
    if T[i] > T2UCL:
        mn1 = mn1 + 1

mn1 = mn1 / 200
print("The false detection rate of PCA T2 is: " + str(mn1))

# 计算误报率和检测率
# 误报率
mn0 = 0
for i in range(0, 200):
    if Q[i] > QUCL:
        mn0 = mn0 + 1

mn0 = mn0 / 200
print("The false alarm rate of PCA Q is: " + str(mn0))

# 检测率
mn1 = 0
for i in range(200, 400):
    if Q[i] > QUCL:
        mn1 = mn1 + 1

mn1 = mn1 / 200
print("The false detection rate of PCA Q is: " + str(mn1))


