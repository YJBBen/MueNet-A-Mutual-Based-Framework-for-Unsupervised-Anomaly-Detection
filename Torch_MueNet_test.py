# -*- coding: utf-8 -*-
import scipy
import torch
import math
import scipy.io as scio
from scipy import stats
import numpy as np
import pandas as pd
from sklearn import preprocessing

import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import torch.nn.functional as F
from collections import Counter
from scipy.stats import entropy
from knncmi.knncmi import cmi
from Loss_function import discretize_data, pad_matrices, ave_joint_entropy, flatten_vectors,split_list

from model_MueNet import Multi_SAE

def average_entropy(vectors,step,n):
    flattened = flatten_vectors(vectors)
    flattened = discretize_data(flattened, math.floor(min(flattened)), math.ceil(max(flattened)), step)
    result = split_list(flattened, n)
    data_D = np.array(result).reshape(-1,n)
    entropy_sum = []
    for i in result:
    # 计算熵
        frequencies = Counter(i)
        probabilities = [f / len(i) for f in frequencies.values()]
        entropy_sum.append(entropy(probabilities))
    return torch.tensor(entropy_sum),data_D
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

def get_index1(lst=None, item=''):
    return [index for (index,value) in enumerate(lst) if value == item]

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

# 载入数据
n_th =200
alpha = 0.95
perf_T2 = np.zeros([2, 15])
perf_Q = np.zeros([2, 15]) # 记录各个尺度的检测性能
data_w = []
step = 0.175

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
X18 = x6 + x11  + 3*x13 + e18 * 0.01
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
X18 = x6 + x11  + 3*x13 + e18 * 0.01
X19 = x2**2+x11**2+e19*0.01
X20 = x1**2+x7**2+x12**2+e20*0.01
Matraix_X16_20_fault1 = np.concatenate((X16,X17,X18,X19,X20),axis=1)
fault1 = np.concatenate((Matraix_X1_5_fault,Matraix_X6_10,Matraix_X11_15,Matraix_X16_20_fault1),axis=1)

#构造异常数据fault2
ramp = np.linspace(0,0.1,200).reshape(-1,1)
print(ramp)
s5_fault =np.concatenate((s5[:200,:] , s5[200:,:]+ramp),axis = 0)
S4_6_fault = np.concatenate((s4,s5_fault,s6),axis=1)
Matraix_X6_10_fault = np.matmul(S4_6_fault,matrix2.T)+0.01*E6_10
x6 = Matraix_X6_10_fault[:,0].reshape(-1,1)
x7 = Matraix_X6_10_fault[:,1].reshape(-1,1)

X16 = x1**2+x2**2+x3**2+e16*0.01
X17 = x1**3+2*x6**2+e17*0.01
X18 = x6 + x11  + 3*x13 + e18 * 0.01
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

######直接引入train时的index
#层次聚类
index1 = [0, 1, 2, 3, 4, 15, 16, 18, 19] #9
index2 = [10, 11, 12, 13, 14, 17] #6
index3 = [5, 6, 7, 8, 9] #5

X_1 = X_test[:, index1]
X_2 = X_test[:, index2]
X_3 = X_test[:, index3]
X_test = np.hstack((X_1,X_2,X_3))

##新增（计算各块的信息熵）
H_data_1,data_1_D = average_entropy(X_1,step,X_1.shape[1])
H_data_2,data_2_D = average_entropy(X_2,step,X_2.shape[1])
H_data_3,data_3_D = average_entropy(X_3,step,X_3.shape[1])

autoencoder_1 = Multi_SAE(len(index1),len(index2),len(index3))

autoencoder_1.load_state_dict(torch.load("D:\Desktop\论文\PAMI\code_to_github\MuSAE_Numerical\model/7-checkpoint.pt"))
# 计算参数数量
total_params = sum(p.numel() for p in autoencoder_1.parameters())
# 将参数数量除以1000，以k为单位表示
total_params_in_k = total_params / 1000
print("Total parameters (in k):{}k".format(total_params_in_k))

data1 = torch.from_numpy(X_1)
data1 = data1.to(torch.float32)
data2 = torch.from_numpy(X_2)
data2 = data2.to(torch.float32)
data3 = torch.from_numpy(X_3)
data3 = data3.to(torch.float32)

autoencoder_1.eval()
# autoencoder_1.eval()
output,feature1,feature2,feature3,output1,output2,output3 = autoencoder_1(data1,data2,data3)

#新增
data1_ = F.softmax(data1, dim=-1)
data2_ = F.softmax(data2, dim=-1)
data3_ = F.softmax(data3, dim=-1)

H_output_1 = F.cross_entropy(output1,data1_,reduction='none') - F.kl_div(F.log_softmax(output1, dim=-1),data1_,reduction='none').mean(dim = 1) ## 使用 mean 来求平均, 维度变为 (960,)
# print("第一块数据的熵为{}".format(H_output_1))
H_output_2 = F.cross_entropy(output2,data2_,reduction='none') - F.kl_div(F.log_softmax(output2, dim=-1),data2_,reduction='none').mean(dim = 1)
# print("第二块数据的熵为{}".format(H_output_2))
H_output_3 = F.cross_entropy(output3,data3_,reduction='none') - F.kl_div(F.log_softmax(output3, dim=-1),data3_,reduction='none').mean(dim = 1)
# print("第三块数据的熵为{}".format(H_output_3))

# #信息熵之差
H_1 = H_data_1 - H_output_1
H_2 = H_data_2 - H_output_2
H_3 = H_data_3 - H_output_3

#长度一致处理（补零）
X_linear_1,X_linear_2,X_linear_3 = pad_matrices(data_1_D,data_2_D,data_3_D)

#计算联合熵
H_in_1_3 = ave_joint_entropy(X_linear_1,X_linear_3)
H_in_2_3 = ave_joint_entropy(X_linear_2,X_linear_3)
H_in_1_2 = ave_joint_entropy(X_linear_1, X_linear_2)


out1_3 = torch.cat((output1, output3), dim=1) #
in1_3 = torch.cat((data1, data3), dim=1)
out2_3 = torch.cat((output2, output3), dim=1)
in2_3 = torch.cat((data2, data3), dim=1)
out1_2 = torch.cat((output1, output2), dim=1)
in1_2 = torch.cat((data1, data2), dim=1)

in1_3_ = F.softmax(in1_3, dim=-1)
in2_3_ = F.softmax(in2_3, dim=-1)
in1_2_ = F.softmax(in1_2, dim=-1)

H_out_1_3 = F.cross_entropy(out1_3,in1_3_,reduction="none") - F.kl_div(F.log_softmax(out1_3, dim=-1),in1_3_,reduction= "none").mean(dim = 1)
# print("第1、3块数据的互信息为{}".format(H_out_1_3))
H_out_2_3 = F.cross_entropy(out2_3,in2_3_,reduction="none") - F.kl_div(F.log_softmax(out2_3, dim=-1),in2_3_,reduction= "none").mean(dim = 1)
# print("第2、3块数据的互信息为{}".format(H_out_2_3))
H_out_1_2 = F.cross_entropy(out1_2, in1_2_, reduction="none") - F.kl_div(F.log_softmax(out1_2, dim=-1), in1_2_, reduction="none").mean(dim = 1)


#通过公式计算互信息H（X）+H(Y)-H（X,Y）
data1_3 = H_data_1 + H_data_3 - H_in_1_3
data2_3 = H_data_2 + H_data_3 - H_in_2_3
data1_2 = H_data_1 + H_data_2 - H_in_1_2
label1_3 = H_output_1 + H_output_3 - H_out_1_3
label2_3 = H_output_2 + H_output_3 - H_out_2_3
label1_2 = H_output_1 + H_output_2 - H_out_1_2
#计算互信息之差
H_MI1_3 = label1_3 - data1_3
H_MI2_3 = label2_3 - data2_3
H_MI1_2 = label1_2 - data1_2
print(H_MI1_3.shape)
# 条件互信息(I(x;y|z))
data_mu = []
for j in range(X_1.shape[0]):
    data_ = np.concatenate((X_linear_1[j,:].reshape(-1,1), X_linear_2[j,:].reshape(-1,1), X_linear_3[j,:].reshape(-1,1)), axis=1)
    # print(data_)
    data_ = pd.DataFrame(data_)
    #计算条件互信息
    data_mu.append(cmi(['0'], ['1'], ['2'], 3,data_))

ave_data_ConMI = torch.tensor(data_mu)
print(ave_data_ConMI.shape)

out1_2_3 = torch.cat((output1, output2, output3), dim=1)
in1_2_3 = torch.cat((data1, data2, data3), dim=1)

in1_2_3_ = F.softmax(in1_2_3,dim=-1)

H_output_123 = F.cross_entropy(out1_2_3, in1_2_3_,reduction="none") - F.kl_div(F.log_softmax(out1_2_3, dim=-1), in1_2_3_,reduction= "none").mean(dim = 1)
# print("第1、2、3块数据的条件互信息为{}".format(H_output_123))
ave_label_ConMI = H_out_2_3-H_output_3+H_out_1_3-H_output_123  # label2_3
CMI = ave_label_ConMI - ave_data_ConMI


feature = torch.cat((feature1, feature2, feature3), 1)
feature = feature.detach().numpy()

X_test_reconstruct = torch.cat((output1,output2,output3),1)
X_test_reconstruct = X_test_reconstruct.detach().numpy()
print(X_test_reconstruct.shape)

T2_1 = np.ones(n_test)
for i in range(n_test):
    a = feature[i, :]
    T2_1[i] = np.dot(a, a.T)
T2_1 = moving_average(T2_1, 5)
th_T2_1 = cal_threshold(T2_1[0:n_th], alpha)
th_T2_1 = th_T2_1 * np.ones(n_test)

# 构造Q统计量
Q_1 = np.ones(n_test)
Q0_1 = X_test - X_test_reconstruct
for k in range(0, 400):   #Numerical 部分实验，用MSEloss时，无0.9*。
    Q_1[k] = 0.5 * np.dot(Q0_1[k, :], Q0_1[k, :].T) + 0.25 * (torch.square(H_1[k]) + torch.square(H_2[k]) + torch.square(H_3[k]))+ 0.25 * (torch.square(H_MI1_3[k]) + torch.square(H_MI2_3[k])+torch.square(H_MI1_2[k])+ torch.square(CMI[k]))
print(Q_1.shape)
Q_1 = moving_average(Q_1, 5)
th_Q_1 = cal_threshold(Q_1[0:n_th], alpha)
th_Q_1 = th_Q_1 * np.ones(n_test)

# 计算性能
FAR_T2_1 = cal_FR(T2_1[0:n_th], th_T2_1)
print("T-A-1:The false alarm rate of SAE T² is: " + str(FAR_T2_1))
FDR_T2_1 = cal_FR(T2_1[n_th:n_test], th_T2_1)
print("T-D-1:The detection rate of SAE T² is: " + str(FDR_T2_1))

print("------------------")

FAR_Q_1 = cal_FR(Q_1[0:n_th], th_Q_1)
print("Q-A-1:The false alarm rate of SAE Q is: " + str(FAR_Q_1))
FDR_Q_1 = cal_FR(Q_1[n_th:n_test], th_Q_1)
print("Q-D-1:The detection rate of SAE Q is: " + str(FDR_Q_1))


# 画图T2
# scipy.io.savemat("./data_mat/MuNet/fault1_T2_data.mat",{"Fdata1T":T2_1})
# scipy.io.savemat("./data_mat/MuNet/fault1_T2_label.mat",{"Flimit1T":th_T2_1})

plt.figure(figsize=(7,5))
plt.subplot(211)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=None, hspace=0.4)
plt.plot(T2_1,'b',lw=1.5,label='mornitoring index')#参数控制颜色和字体的粗细
plt.plot(th_T2_1,'r',label='control limit')
plt.grid(True)
plt.legend(loc = 0,fontsize = 14)
# plt.title('Mornitoring performance')
plt.xlabel('Sample number',fontsize = 14)
plt.ylabel('T²',fontsize = 14)

# 画图Q
# scipy.io.savemat("./data_mat/MuNet/fault1_Q_data.mat",{"Fdata1Q":Q_1})
# scipy.io.savemat("./data_mat/MuNet/fault1_Q_label.mat",{"Flimit1Q":th_Q_1})
plt.subplot(212)
plt.plot(Q_1,'b',lw=1.5,label='mornitoring index')#参数控制颜色和字体的粗细
plt.plot(th_Q_1,'r',label='control limit')
plt.grid(True)
plt.legend(loc = 0,fontsize = 14)
plt.title('Mornitoring performance',fontsize = 20)
plt.xlabel('Sample number',fontsize = 14)
plt.ylabel('SPE',fontsize = 14)
plt.show()