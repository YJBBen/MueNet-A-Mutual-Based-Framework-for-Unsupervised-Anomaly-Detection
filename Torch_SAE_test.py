# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 15:33:58 2019

@author: Administrator
"""
import scipy
import scipy.io as scio
import numpy as np
import torch
from sklearn import preprocessing
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics as ms
import pandas as pd
import torch.nn as nn


def moving_average(l, N):
    sum = 0
    result = list( 0 for x in l)
    for i in range( 0, N ):
        sum = sum + l[i]
        result[i] = sum / (i+1)
    for i in range( N, len(l) ):
        sum = sum - l[i-N] + l[i]
        result[i] = sum / N
    return np.array(result)
# 定义函数
def cal_threshold(x, alpha):
    kernel = stats.gaussian_kde(x)
    step = np.linspace(0,100,10000)
    pdf = kernel(step)
    for i in range(len(step)):
        if sum(pdf[0:(i+1)]) / sum(pdf) > alpha:
            break
    return step[i+1]

# 计算检测性能
def cal_FR(statistics, limit):
    mn = 0
    FR = 0
    for i in range(len(statistics)):
        if statistics[i] > limit[0]:
            mn = mn+1
        FR = mn/len(statistics)
    return FR

n_th = 200
alpha = 0.95
n_th =200
alpha = 0.95
perf_T2 = np.zeros([2, 15])
perf_Q = np.zeros([2, 15]) # 记录各个尺度的检测性能


n = np.arange(1,401)
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
s1 = s1[:,np.newaxis]
s2 = np.random.normal(mu1, sigma1, sampleNo)
s2 = s2[:,np.newaxis]
s3 = np.random.normal(mu1, sigma1, sampleNo)
s3 = s3[:,np.newaxis]
s4 = np.random.normal(mu2, sigma2, sampleNo)
s4 = s4[:,np.newaxis]
s5 = np.random.normal(mu2, sigma2, sampleNo)
s5 = s5[:,np.newaxis]
s6 = np.random.normal(mu2, sigma2, sampleNo)
s6 = s6[:,np.newaxis]
s7 = np.random.normal(mu3, sigma3, sampleNo)
s7 = s7[:,np.newaxis]
s8 = np.random.normal(mu3, sigma3, sampleNo)
s8 = s8[:,np.newaxis]
s9 = np.random.normal(mu3, sigma3, sampleNo)
s9 = s9[:,np.newaxis]


e1 = np.random.normal(mu_e1, sigma_e1, sampleNo)
e1 = e1[:,np.newaxis]
e2 = np.random.normal(mu_e1, sigma_e1, sampleNo)
e2 = e2[:,np.newaxis]
e3 = np.random.normal(mu_e1, sigma_e1, sampleNo)
e3 = e3[:,np.newaxis]
e4 = np.random.normal(mu_e1, sigma_e1, sampleNo)
e4 = e4[:,np.newaxis]
e5 = np.random.normal(mu_e1, sigma_e1, sampleNo)
e5 = e5[:,np.newaxis]
e6 = np.random.normal(mu_e1, sigma_e1, sampleNo)
e6 = e6[:,np.newaxis]
e7 = np.random.normal(mu_e1, sigma_e1, sampleNo)
e7 = e7[:,np.newaxis]
e8 = np.random.normal(mu_e1, sigma_e1, sampleNo)
e8 = e8[:,np.newaxis]
e9 = np.random.normal(mu_e1, sigma_e1, sampleNo)
e9 = e9[:,np.newaxis]
e10 = np.random.normal(mu_e1, sigma_e1, sampleNo)
e10 = e10[:,np.newaxis]
e11 = np.random.normal(mu_e1, sigma_e1, sampleNo)
e11 = e11[:,np.newaxis]
e12 = np.random.normal(mu_e1, sigma_e1, sampleNo)
e12 = e12[:,np.newaxis]
e13 = np.random.normal(mu_e1, sigma_e1, sampleNo)
e13 = e13[:,np.newaxis]
e14 = np.random.normal(mu_e1, sigma_e1, sampleNo)
e14 = e14[:,np.newaxis]
e15 = np.random.normal(mu_e1, sigma_e1, sampleNo)
e15 = e15[:,np.newaxis]
e16 = np.random.normal(mu_e1, sigma_e1, sampleNo)
e16 = e16[:,np.newaxis]
e17 = np.random.normal(mu_e1, sigma_e1, sampleNo)
e17 = e17[:,np.newaxis]
e18 = np.random.normal(mu_e1, sigma_e1, sampleNo)
e18 = e18[:,np.newaxis]
e19 = np.random.normal(mu_e1, sigma_e1, sampleNo)
e19 = e19[:,np.newaxis]
e20 = np.random.normal(mu_e1, sigma_e1, sampleNo)
e20 = e20[:,np.newaxis]
# print(s6.shape)
S1_3 = np.concatenate((s1,s2,s3),axis=1)
S4_6 = np.concatenate((s4,s5,s6),axis=1)
S7_9 = np.concatenate((s7,s8,s9),axis=1)
E1_5 = np.concatenate((e1,e2,e3,e4,e5),axis=1)
E6_10 = np.concatenate((e6,e7,e8,e9,e10),axis=1)
E11_15 = np.concatenate((e11,e12,e13,e14,e15),axis=1)

matrix1 = np.array([[1.57,1.37,1.80],[1.73,1.05,1.70],[1.82,1.40,1.60],[1.65,1.20,1.50],[1.47,1.24,1.60]])
matrix2 = np.array([[1.67,1.47,1.70],[1.63,1.15,1.80],[1.72,1.30,1.70],[1.55,1.30,1.60],[1.45,1.38,1.80]])
matrix3 = np.array([[1.58,1.92,1.47],[1.53,1.20,1.26],[1.45,1.53,1.79],[1.86,1.76,1.89],[1.77,1.73,1.53]])

Matraix_X1_5 = np.matmul(S1_3,matrix1.T)+0.01*E1_5
Matraix_X6_10 = np.matmul(S4_6,matrix2.T)+0.01*E6_10
Matraix_X11_15 = np.matmul(S7_9,matrix3.T)+0.01*E11_15
#耦合数据
x1 = Matraix_X1_5[:,0].reshape(-1,1)
x2 = Matraix_X1_5[:,1].reshape(-1,1)
x3 = Matraix_X1_5[:,2].reshape(-1,1)
x6 = Matraix_X6_10[:,0].reshape(-1,1)
x7 = Matraix_X6_10[:,1].reshape(-1,1)
x11 = Matraix_X11_15[:,0].reshape(-1,1)
x12 = Matraix_X11_15[:,1].reshape(-1,1)
x13 = Matraix_X11_15[:,2].reshape(-1,1)
X16 = x1**2+x2**2+x3**2+e16*0.01
X17 = x1**3+2*x6**2+e17*0.01
X18 = x6  + x11  + 3*x13 + e18 * 0.01
X19 = x2**2+x11**2+e19*0.01
X20 = x1**2+x7**2+x12**2+e20*0.01
Matraix_X16_20 = np.concatenate((X16,X17,X18,X19,X20),axis=1)
data_nomal = np.concatenate((Matraix_X1_5,Matraix_X6_10,Matraix_X11_15,Matraix_X16_20),axis=1)


#构造异常数据fault1
s1_fault =np.concatenate((s1[:200,:] , s1[200:,:]+0.1),axis = 0)
S1_3_fault = np.concatenate((s1_fault,s2,s3),axis=1)
Matraix_X1_5_fault = np.matmul(S1_3_fault,matrix1.T)+0.01*E1_5
x1 = Matraix_X1_5_fault[:,0].reshape(-1,1)
x2 = Matraix_X1_5_fault[:,1].reshape(-1,1)
x3 = Matraix_X1_5_fault[:,2].reshape(-1,1)
X16 = x1**2+x2**2+x3**2+e16*0.01
X17 = x1**3+2*x6**2+e17*0.01
X18 = x6  + x11  + 3*x13 + e18 * 0.01
X19 = x2**2+x11**2+e19*0.01
X20 = x1**2+x7**2+x12**2+e20*0.01
Matraix_X16_20_fault1 = np.concatenate((X16,X17,X18,X19,X20),axis=1)
fault1 = np.concatenate((Matraix_X1_5_fault,Matraix_X6_10,Matraix_X11_15,Matraix_X16_20_fault1),axis=1)

#构造异常数据fault2
ramp = np.linspace(0,0.1,200).reshape(-1,1)
# # # print(ramp.shape)
s5_fault =np.concatenate((s5[:200,:] , s5[200:,:]+ramp),axis = 0)
S4_6_fault = np.concatenate((s4,s5_fault,s6),axis=1)
Matraix_X6_10_fault = np.matmul(S4_6_fault,matrix2.T)+0.01*E6_10
x6 = Matraix_X6_10_fault[:,0].reshape(-1,1)
x7 = Matraix_X6_10_fault[:,1].reshape(-1,1)
X16 = x1**2+x2**2+x3**2+e16*0.01
X17 = x1**3+2*x6**2+e17*0.01
X18 = x6  + x11  + 3*x13 + e18 * 0.01
X19 = x2**2+x11**2+e19*0.01
X20 = x1**2+x7**2+x12**2+e20*0.01
Matraix_X16_20_fault2 = np.concatenate((X16,X17,X18,X19,X20),axis=1)
fault2 = np.concatenate((Matraix_X1_5,Matraix_X6_10_fault,Matraix_X11_15,Matraix_X16_20_fault2),axis=1)
print(fault2.shape)
scaler = preprocessing.StandardScaler().fit(data_nomal)# 创建标准化转换器
X_training = preprocessing.scale(data_nomal)# 标准化处理
n_training = X_training.shape[0]
m_training = X_training.shape[1]


data_testing = scaler.transform(fault2)# 测试数据标准化
X_test = data_testing
n_test = X_test.shape[0]
m_test = X_test.shape[1]

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

X_test = torch.from_numpy(X_test)
X_test = X_test.to(torch.float32)
autoencoder = Net()
autoencoder.load_state_dict(torch.load("D:\Desktop\Block_att/1-数值仿真_New_test_不统一维度\整体实验/1-4checkpoint.pt"))
# 计算参数数量
total_params = sum(p.numel() for p in autoencoder.parameters())
# 将参数数量除以1000，以k为单位表示
total_params_in_k = total_params / 1000
print("Total parameters (in k):{}k".format(total_params_in_k))

feature,X_test_reconstruct = autoencoder(X_test)

feature = feature.detach().numpy()
X_test_reconstruct = X_test_reconstruct.detach().numpy()

# ---------------------------------------------------

# 构造T2统计量

T2 = np.ones(n_test)
for i in range(n_test):
    a = feature[i, :]
    T2[i] = np.dot(a, a.T)
# T2 = moving_average(T2, 5)

th_T2 = cal_threshold(T2[0:n_th], alpha)
th_T2 = th_T2*np.ones(n_test)

# 构造Q统计量
Q = np.ones(n_test)
Q0=X_test-X_test_reconstruct
for i in range(0,400):
    t1=Q0[i,:].reshape((len(Q0[i,:]),1))
    t2=Q0[i,:].reshape((1,len(Q0[i,:])))
    Q[i] = np.dot(t2,t1)

# Q = moving_average(Q, 5)
th_Q = cal_threshold(Q[0:n_th], alpha)
th_Q = th_Q*np.ones(n_test)

#计算性能
FAR_T2 = cal_FR(T2[0:200], th_T2)
print ("The false alarm rate of SAE T² is: " + str(FAR_T2))
FDR_T2 = cal_FR(T2[200:], th_T2)
print ("The detection rate of SAE T² is: " + str(FDR_T2))
FAR_Q = cal_FR(Q[0:200], th_Q)
print ("The false alarm rate of SAE Q is: " + str(FAR_Q))
FDR_Q = cal_FR(Q[200:], th_Q)
print ("The detection rate of SAE Q is: " + str(FDR_Q))


# 画图T2
# scipy.io.savemat("./data_mat/SAE/fault2_T2_data.mat",{"SAEdata2T":T2})
# scipy.io.savemat("./data_mat/SAE/fault2_T2_label.mat",{"SAElimit2T":th_T2})
plt.figure(figsize=(7,5))
plt.subplot(211)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=None, hspace=0.4)
plt.plot(T2,'b',lw=1.5,label='mornitoring index:T²')#参数控制颜色和字体的粗细
plt.plot(th_T2,'r',label='control limit')
plt.grid(True)
plt.legend(loc = 0,fontsize = 14)
plt.title('Mornitoring performance of SAE')
plt.xlabel('Sample number',fontsize = 14)
plt.ylabel('T²',fontsize = 14)



# 画图Q
# scipy.io.savemat("./data_mat/SAE/fault2_Q_data.mat",{"SAEdata2Q":Q})
# scipy.io.savemat("./data_mat/SAE/fault2_Q_label.mat",{"SAElimit2Q":th_Q})
#plt.figure(figsize=(7,4))
plt.subplot(212)
plt.plot(Q,'b',lw=1.5,label='mornitoring index')#参数控制颜色和字体的粗细
plt.plot(th_Q,'r',lw=1,label='control limit')
plt.grid(True)
plt.legend(loc = 0,fontsize = 14)
plt.title('Mornitoring performance of SAE',fontsize = 20)
plt.xlabel('Sample number',fontsize = 14)
plt.ylabel('SPE',fontsize = 14)
plt.show()


