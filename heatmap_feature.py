# 绘制混淆矩阵

import torch
import math
import scipy
import numpy as np
import scipy.io as scio
import matplotlib as mpl
from sklearn import preprocessing
from scipy import stats
from knncmi.knncmi import cmi
import matplotlib.pyplot as plt
from collections import Counter
import torch.nn.functional as F
from scipy.stats import entropy
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.cluster import AgglomerativeClustering
from Loss_function import discretize_data, pad_matrices, ave_joint_entropy, flatten_vectors, \
    split_list
from model_MueNet import Multi_SAE

def plot_confusion_matrix(conf_matrix):
    colors = ['#AADDFF', '#99CCFF', '#88BBEE', '#77AAEE']
    cmap = mpl.colors.ListedColormap(colors)
    plt.imshow(conf_matrix, cmap)
    indices = range(conf_matrix.shape[0])
    labels = [1, 2, 3]
    plt.xticks(indices, labels, fontsize=25)
    plt.yticks(indices, labels, fontsize=25)
    plt.rcParams['font.size'] = 16
    plt.colorbar()
    plt.xlabel('', fontdict={'family': 'Times New Roman', 'size': 25})
    plt.ylabel('', fontdict={'family': 'Times New Roman', 'size': 25})
    # 显示数据
    for first_index in range(conf_matrix.shape[0]):
        for second_index in range(conf_matrix.shape[1]):
            plt.text(first_index, second_index, conf_matrix[second_index, first_index], horizontalalignment='center',
                     fontdict={'family': 'Times New Roman', 'size': 25})
    plt.savefig('heatmap_confusion_matrix.jpg')
    plt.show()

def get_index1(lst=None, item=''):
    return [index for (index, value) in enumerate(lst) if value == item]

def average_entropy(vectors, step, n):
    # 展平向量集合
    # if isinstance(vectors,np.ndarray):
    flattened = flatten_vectors(vectors)
    # print(flattened)
    flattened = discretize_data(flattened, math.floor(min(flattened)), math.ceil(max(flattened)), step)
    result = split_list(flattened, n)  # 将展开后的列表切成len(flattened)/n  段
    data_D = np.array(result).reshape(-1, n)
    entropy_sum = []
    for i in result:
        # 计算熵
        frequencies = Counter(i)
        probabilities = [f / len(i) for f in frequencies.values()]
        entropy_sum.append(entropy(probabilities))

    return torch.tensor(entropy_sum), data_D  # .reshape(-1,1)


# 定义函数
def cal_threshold(x, alpha):
    kernel = stats.gaussian_kde(x)
    step = np.linspace(0, 100, 10000)
    pdf = kernel(step)
    for i in range(len(step)):
        if sum(pdf[0:(i + 1)]) / sum(pdf) > alpha:
            break
    return step[i + 1]


# 计算检测性能
def cal_FR(statistics, limit):
    mn = 0
    FR = 0
    for i in range(len(statistics)):
        if statistics[i] > limit[0]:
            mn = mn + 1
        FR = mn / len(statistics)
    return FR


def get_index1(lst=None, item=''):
    return [index for (index, value) in enumerate(lst) if value == item]


def moving_average(l, N):
    sum = 0
    result = list(0 for x in l)
    for i in range(0, N):
        sum = sum + l[i]
        result[i] = sum / (i + 1)
    for i in range(N, len(l)):
        sum = sum - l[i - N] + l[i]
        result[i] = sum / N
    return np.array(result)


# 载入数据
n_th = 200
alpha = 0.95
perf_T2 = np.zeros([2, 15])
perf_Q = np.zeros([2, 15])  # 记录各个尺度的检测性能
data_w = []
step = 0.175

n = np.arange(1, 401)
sampleNo = 400
mu1 = 0
sigma1 = 0.1
mu2 = 0
sigma2 = 0.16
mu3 = 0
sigma3 = 0.18
mu_e1 = 0
sigma_e1 = 0.1414  # 方差为0.02
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

# data_training_all = np.hstack((data_training[:,0:22],data_training[:,41:52]))# 提取33个变量
scaler = preprocessing.StandardScaler().fit(data_nomal)  # 创建标准化转换器
X_training = preprocessing.scale(data_nomal)  # 标准化处理

# 层次聚类
index1 = [0, 1, 2, 3, 4, 15, 16, 18, 19]  # 9
index2 = [10, 11, 12, 13, 14, 17]  # 6
index3 = [5, 6, 7, 8, 9]  # 5

X_1 = X_training[:, index1]
X_2 = X_training[:, index2]
X_3 = X_training[:, index3]
X_training = np.hstack((X_1, X_2, X_3))

##新增（计算各块的信息熵）
H_data_1, data_1_D = average_entropy(X_1, step, X_1.shape[1])
H_data_2, data_2_D = average_entropy(X_2, step, X_2.shape[1])
H_data_3, data_3_D = average_entropy(X_3, step, X_3.shape[1])

# 改为SAE_CL
# SAE = load_model('SAE.h5')
# SAE_encoder = load_model('SAE_encoder.h5')
autoencoder_1 = Multi_SAE(len(index1), len(index2), len(index3))
# print("x1 lie:{}".format(X_1.shape[1]))


autoencoder_1.load_state_dict(torch.load("D:\Desktop\Block_att/1-数值仿真_New_test_不统一维度/Matrix-checkpoint.pt"))
# 计算参数数量
total_params = sum(p.numel() for p in autoencoder_1.parameters())
# 将参数数量除以1000，以k为单位表示
total_params_in_k = total_params / 1000
print("Total parameters (in k):{}k".format(total_params_in_k))
# autoencoder_2.load_state_dict(torch.load("./性能汇总/K-bMOM-模型参数50-/model2_checkpoint.pt"))
# autoencoder_3.load_state_dict(torch.load("./性能汇总/K-bMOM-模型参数50-/model3_checkpoint.pt"))

data1 = torch.from_numpy(X_1)
data1 = data1.to(torch.float32)
data2 = torch.from_numpy(X_2)
data2 = data2.to(torch.float32)
data3 = torch.from_numpy(X_3)
data3 = data3.to(torch.float32)

autoencoder_1.eval()
# autoencoder_1.eval()
output, feature1, feature2, feature3, output1, output2, output3 = autoencoder_1(data1, data2, data3)

feature11 = feature1.squeeze(0)
feature22 = feature2.squeeze(0)
feature33 = feature3.squeeze(0)

X_1 = feature11
X_2 = feature22
X_3 = feature33

Q_1 = F.softmax(feature11, dim=-1)
Q_2 = F.softmax(feature22, dim=-1)
Q_3 = F.softmax(feature33, dim=-1)

X_1_2 = torch.cat((X_1, X_2), dim=1)
X_1_3 = torch.cat((X_1, X_3), dim=1)

X_2_3 = torch.cat((X_2, X_3), dim=1)

Q_1_2 = F.softmax(X_1_2, dim=-1)
Q_1_3 = F.softmax(X_1_3, dim=-1)

Q_2_3 = F.softmax(X_2_3, dim=-1)

H_1 = torch.mean(-Q_1 * torch.log(Q_1), dim=1) * X_1.shape[1]
# print(H_1)
H_2 = torch.mean(-Q_2 * torch.log(Q_2), dim=1) * X_2.shape[1]
# print(H_2)
H_3 = torch.mean(-Q_3 * torch.log(Q_3), dim=1) * X_3.shape[1]

# print(H_3)
H_1_2 = torch.mean(-Q_1_2 * torch.log(Q_1_2), dim=1) * (X_1.shape[1] + X_2.shape[1])
H_1_3 = torch.mean(-Q_1_3 * torch.log(Q_1_3), dim=1) * (X_1.shape[1] + X_3.shape[1])

H_2_3 = torch.mean(-Q_2_3 * torch.log(Q_2_3), dim=1) * (X_2.shape[1] + X_3.shape[1])

# print(H_1_2)
M1_2 = H_1 + H_2 - H_1_2
M1_3 = H_1 + H_3 - H_1_3

M2_3 = H_2 + H_3 - H_2_3

# print(M1_2)
M1_2 = torch.mean(M1_2).detach().numpy()
M1_3 = torch.mean(M1_3).detach().numpy()

M2_3 = torch.mean(M2_3).detach().numpy()

M1_2 = np.around(M1_2, decimals=2)
M1_3 = np.around(M1_3, decimals=2)

M2_3 = np.around(M2_3, decimals=2)

block_1_1 = np.around(torch.mean(H_1).detach().numpy(), decimals=2)
block_2_2 = np.around(torch.mean(H_2).detach().numpy(), decimals=2)
block_3_3 = np.around(torch.mean(H_3).detach().numpy(), decimals=2)

block_1_2 = M1_2
block_1_3 = M1_3

block_2_3 = M2_3
print(block_1_3)
list = [block_1_1, block_1_2, block_1_3, block_2_2, block_2_3, block_3_3]
matrix = np.zeros((3, 3))
for i in range(matrix.shape[0]):
    for j in range(i, matrix.shape[1]):
        if i == 0:
            matrix[i][j] = round(list[i + j], 2)
        if i == 1:
            matrix[i][j] = round(list[i + j + 1], 2)
        if i == 2:
            matrix[i][j] = round(list[i + j + 1], 2)
        if i == 3:
            matrix[i][j] = round(list[i + j + 3], 2)

matrix += matrix.T - np.diag(matrix.diagonal())

print(matrix)
matrix = np.round(matrix, decimals=2)

plot_confusion_matrix(matrix)
