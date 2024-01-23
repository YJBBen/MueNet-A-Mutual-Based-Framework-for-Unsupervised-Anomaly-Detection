import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
import pandas as pd
from knncmi.knncmi import cmi

warnings.filterwarnings('ignore')
import math
from collections import Counter
from scipy.stats import entropy


def flatten_vectors(vectors):
    flattened = []
    for vec in vectors:
        flattened.extend(vec)
    return flattened


def split_list(lst, n):
    # 展平向量集合
    return [lst[i:i + n] for i in range(0, len(lst), n)]


def average_entropy(vectors, step, n):
    # 展平向量集合
    if isinstance(vectors, np.ndarray):
        flattened = flatten_vectors(vectors)
        flattened = discretize_data(flattened, math.floor(min(flattened)), math.ceil(max(flattened)), step)
    result = split_list(flattened, n)  # 将展开后的列表切成len(flattened)/n  段
    data_D = np.array(result).reshape(-1, n)
    entropy_ = 0
    for i in result:
        # 计算熵
        frequencies = Counter(i)
        probabilities = [f / len(i) for f in frequencies.values()]
        entropy_sum = entropy(probabilities)
        entropy_ += entropy_sum
    return entropy_ / len(result), data_D


def ave_joint_entropy(x, y):
    """
    计算两个维度相同向量x和y的联合熵
    """
    assert x.shape[1] == y.shape[1], "x和y的列数必须相等"
    joint_entropy = 0
    for row in range(x.shape[0]):
        xy = list(zip(x[row, :], y[row, :]))  # 将x和y按行组合成二维列表
        counts = {}  # 统计每个元素的出现次数
        for i, j in xy:
            if (i, j) in counts:
                counts[(i, j)] += 1
            else:
                counts[(i, j)] = 1
        joint_probs = [count / len(xy) for count in counts.values()]  # 计算联合概率
        entropy_j = entropy(joint_probs)
        joint_entropy += entropy_j  # 计算联合熵
    ave_joint_entropy = joint_entropy / x.shape[0]
    return ave_joint_entropy


# data.shape[0]

# #数据等距离散化处理
def discretize_data(data, lower, upper, step):
    """
    对给定的一维numpy数组进行区间离散化

    参数：
    data: 一维numpy数组，需要进行离散化的数据
    lower: 离散化区间的下界
    upper: 离散化区间的上界
    step: 离散化区间的间隔

    返回：
    离散化后的标签，为numpy数组类型
    """
    # 生成分割点
    bins = np.arange(lower, upper + step, step)

    # 离散化数据并指定标签
    labels = bins[:-1] + step / 2
    discretized_data = pd.cut(data, bins=bins, labels=labels)
    #     print(discretize_data)
    # 将Series对象转换为numpy数组
    discretized_data = discretized_data.to_numpy()

    return discretized_data


# 按列补零
def pad_matrices(mat1, mat2, mat3):
    # 计算三个矩阵的列数
    cols1, cols2, cols3 = mat1.shape[1], mat2.shape[1], mat3.shape[1]
    # 找到最大的列数
    max_cols = max(cols1, cols2, cols3)
    # 对第一个矩阵进行补零
    pad_width = ((0, 0), (0, max_cols - cols1))
    mat1_padded = np.pad(mat1, pad_width, mode='constant', constant_values=0)
    # 对第二个矩阵进行补零
    pad_width = ((0, 0), (0, max_cols - cols2))
    mat2_padded = np.pad(mat2, pad_width, mode='constant', constant_values=0)
    # 对第三个矩阵进行补零
    pad_width = ((0, 0), (0, max_cols - cols3))
    mat3_padded = np.pad(mat3, pad_width, mode='constant', constant_values=0)

    return mat1_padded, mat2_padded, mat3_padded


class SAE_loss_CL(nn.Module):
    def __init__(self, c=3, step=0.01):
        super(SAE_loss_CL, self).__init__()
        self.c = c
        self.step = step

    def forward(self, data, data1, data2, data3, output1, output2, output3, output):
        certerion = nn.MSELoss(reduction="mean")
        MSE_loss = 1 / 2 * (certerion(data, output))

        data_1 = data1.detach().numpy()
        data_2 = data2.detach().numpy()
        data_3 = data3.detach().numpy()

        # 计算各块的信息熵
        H_data_1, data_1_D = average_entropy(data_1, self.step, data_1.shape[1])
        H_data_2, data_2_D = average_entropy(data_2, self.step, data_2.shape[1])
        H_data_3, data_3_D = average_entropy(data_3, self.step, data_3.shape[1])

        data1_ = F.softmax(data1, dim=-1)
        data2_ = F.softmax(data2, dim=-1)
        data3_ = F.softmax(data3, dim=-1)

        H_output_1 = F.cross_entropy(output1, data1_, reduction="mean") - F.kl_div(F.log_softmax(output1, dim=-1),
                                                                                   data1_, reduction="mean")
        # print("第一块数据的熵为{}".format(H_output_1))
        H_output_2 = F.cross_entropy(output2, data2_, reduction="mean") - F.kl_div(F.log_softmax(output2, dim=-1),
                                                                                   data2_, reduction="mean")
        # print("第二块数据的熵为{}".format(H_output_2))
        H_output_3 = F.cross_entropy(output3, data3_, reduction="mean") - F.kl_div(F.log_softmax(output3, dim=-1),
                                                                                   data3_, reduction="mean")
        # print("第三块数据的熵为{}".format(H_output_3))
        # #信息熵之差
        H_1 = H_data_1 - H_output_1
        H_2 = H_data_2 - H_output_2
        H_3 = H_data_3 - H_output_3

        # 长度一致处理（补零）
        X_linear_1, X_linear_2, X_linear_3 = pad_matrices(data_1_D, data_2_D, data_3_D)

        # 计算联合熵
        H_in_1_3 = ave_joint_entropy(X_linear_1, X_linear_3)
        H_in_2_3 = ave_joint_entropy(X_linear_2, X_linear_3)
        H_in_1_2 = ave_joint_entropy(X_linear_1, X_linear_2)

        out1_3 = torch.cat((output1, output3), dim=1)  #
        in1_3 = torch.cat((data1, data3), dim=1)
        out2_3 = torch.cat((output2, output3), dim=1)
        in2_3 = torch.cat((data2, data3), dim=1)
        out1_2 = torch.cat((output1, output2), dim=1)
        in1_2 = torch.cat((data1, data2), dim=1)

        in1_3_ = F.softmax(in1_3, dim=-1)
        in2_3_ = F.softmax(in2_3, dim=-1)
        in1_2_ = F.softmax(in1_2, dim=-1)

        H_out_1_3 = F.cross_entropy(out1_3, in1_3_, reduction="mean") - F.kl_div(F.log_softmax(out1_3, dim=-1), in1_3_,
                                                                                 reduction="mean")
        # print("第1、3块数据的互信息为{}".format(H_out_1_3))
        H_out_2_3 = F.cross_entropy(out2_3, in2_3_, reduction="mean") - F.kl_div(F.log_softmax(out2_3, dim=-1), in2_3_,
                                                                                 reduction="mean")
        # print("第2、3块数据的互信息为{}".format(H_out_2_3))
        H_out_1_2 = F.cross_entropy(out1_2, in1_2_, reduction="mean") - F.kl_div(F.log_softmax(out1_2, dim=-1), in1_2_,
                                                                                 reduction="mean")

        # 计算互信息H（X）+H(Y)-H（X,Y）
        data1_3 = H_data_1 + H_data_3 - H_in_1_3
        data2_3 = H_data_2 + H_data_3 - H_in_2_3
        data1_2 = H_data_1 + H_data_2 - H_in_1_2
        label1_3 = H_output_1 + H_output_3 - H_out_1_3
        label2_3 = H_output_2 + H_output_3 - H_out_2_3
        label1_2 = H_output_1 + H_output_2 - H_out_1_2
        # 计算互信息之差
        H_MI1_3 = label1_3 - data1_3
        H_MI2_3 = label2_3 - data2_3
        H_MI1_2 = label1_2 - data1_2

        # 条件互信息
        data_mu = 0
        for i in range(data_1.shape[0]):
            data_ = np.concatenate(
                (X_linear_1[i, :].reshape(-1, 1), X_linear_2[i, :].reshape(-1, 1), X_linear_3[i, :].reshape(-1, 1)),
                axis=1)
            # print(data_)
            data_ = pd.DataFrame(data_)
            # 计算条件互信息
            data_mu += cmi(['0'], ['1'], ['2'], 3, data_)

        ave_data_ConMI = data_mu

        out1_2_3 = torch.cat((output1, output2, output3), dim=1)
        in1_2_3 = torch.cat((data1, data2, data3), dim=1)
        in1_2_3_ = F.softmax(in1_2_3, dim=-1)
        H_output_123 = F.cross_entropy(out1_2_3, in1_2_3_, reduction="mean") - F.kl_div(F.log_softmax(out1_2_3, dim=-1),
                                                                                        in1_2_3_, reduction="mean")

        ave_label_ConMI = H_out_2_3 - H_output_3 + H_out_1_3 - H_output_123
        CMI = ave_label_ConMI - ave_data_ConMI

        return 0.5 * MSE_loss + 0.25 * (torch.square(H_1) + torch.square(H_2) + torch.square(H_3)) + 0.25 * (
                    torch.square(H_MI1_3) + torch.square(H_MI2_3) + torch.square(H_MI1_2) + torch.square(CMI))
