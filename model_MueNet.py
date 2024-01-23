import torch
import torch.nn.functional as F
import torch.nn as nn


class MultiheadAttention(nn.Module):
    # n_heads：多头注意力的数量
    # hid_dim：每个词输出的向量维度
    def __init__(self, hid_dim1, hid_dim, n_heads, dropout):
        super(MultiheadAttention, self).__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads

        # 强制 hid_dim 必须整除 h
        assert hid_dim % n_heads == 0
        # 定义 W_q 矩阵
        self.w_q = nn.Linear(hid_dim1, hid_dim1)
        # 定义 W_k 矩阵
        self.w_k = nn.Linear(hid_dim1, hid_dim1)
        # 定义 W_v 矩阵
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        # 缩放
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads]))

    def forward(self, query, key, value):
        bsz = query.shape[0]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        v = torch.zeros((bsz, V.shape[2]))
        for i in range(Q.shape[1]):
            Q_i = F.softmax(Q[:, i, :], dim=-1)
            H_Q = torch.unsqueeze(torch.mean(Q_i * torch.log_softmax(Q[:, i, :], dim=-1), dim=1),
                                  dim=1)
            for j in range(K.shape[1]):
                K_j = F.softmax(K[:, j, :], dim=-1)
                H_K = torch.unsqueeze(torch.mean(K_j * torch.log_softmax(K[:, j, :], dim=-1), dim=1), dim=1)

                Q_kj = torch.cat((Q[:, i, :], K[:, j, :]), dim=1)
                Q_kj = F.softmax(Q_kj, dim=-1)

                H_Q_K = torch.unsqueeze(torch.mean(Q_kj * torch.log_softmax(Q_kj, dim=-1), dim=1), dim=1)
                I_Q_K1 = -(H_Q + H_K - H_Q_K) / self.scale
                if j == 0:
                    I_Q_K = I_Q_K1
                else:
                    I_Q_K = torch.cat((I_Q_K, I_Q_K1), dim=1)

            I_Q_K = torch.softmax(I_Q_K, dim=-1)  # 按列维度

            for m in range(V.shape[1]):
                v += I_Q_K[:, m].reshape(-1, 1) * V[:, m, :]
            v1 = torch.unsqueeze(v, dim=1)

            if i == 0:
                V_new = v1
            else:
                V_new = torch.cat([V_new, v1], dim=1)

        x = self.fc(V_new)
        return x

# 模型最好都在__init__中定义好，在forward中直接调用
class Multi_SAE(nn.Module):
    def __init__(self, input_size1, input_size2, input_size3):
        super(Multi_SAE, self).__init__()
        # input = []
        self.input1 = input_size1
        self.input2 = input_size2
        self.input3 = input_size3
        self.att_all = 10  # 隐变量统一维度，用于multihead
        self.att_out = self.att_all * 3

        self.encoder1_1 = nn.Sequential(
            nn.Linear(self.input1, 16),
            nn.Tanh())
        self.encoder1_2 = nn.Sequential(
            nn.Linear(16, 11),
            nn.Tanh())
        self.att_1 = nn.Sequential(
            nn.Linear(11, 6)
        )

        self.encoder2_1 = nn.Sequential(
            nn.Linear(self.input2, 10),
            nn.Tanh())
        self.encoder2_2 = nn.Sequential(
            nn.Linear(10, 6),
            nn.Tanh())
        self.att_2 = nn.Sequential(
            nn.Linear(6, 4)
        )

        self.encoder3_1 = nn.Sequential(
            nn.Linear(self.input3, 9),
            nn.Tanh())
        self.encoder3_2 = nn.Sequential(
            nn.Linear(9, 5),
            nn.Tanh())
        self.att_3 = nn.Sequential(
            nn.Linear(5, 3)
        )
        self.all_cat_B1 = nn.Sequential(
            nn.Linear(33, self.att_all)
        )
        self.all_cat_B2 = nn.Sequential(
            nn.Linear(20, self.att_all)
        )
        self.all_cat_B3 = nn.Sequential(
            nn.Linear(17, self.att_all)
        )

        self.decoder1_1 = nn.Sequential(
            nn.Linear(self.att_out, 11),
            nn.Tanh(),

            nn.Linear(11, 16),
            nn.Tanh(),
            nn.Linear(16, self.input1),

        )
        self.decoder2_1 = nn.Sequential(
            nn.Linear(self.att_out, 6),
            nn.Tanh(),
            nn.Linear(6, 10),
            nn.Tanh(),
            nn.Linear(10, self.input2),

        )
        self.decoder3_1 = nn.Sequential(
            nn.Linear(self.att_out, 5),
            nn.Tanh(),
            nn.Linear(5, 9),
            nn.Tanh(),
            nn.Linear(9, self.input3),

        )
        self.linear1 = nn.Sequential(nn.Linear(self.att_all * 2, self.att_all))
        self.attention_block_1 = MultiheadAttention(hid_dim1=self.att_all, hid_dim=self.att_all, n_heads=2,
                                                    dropout=0.1)
        self.attention_block_2 = MultiheadAttention(hid_dim1=self.att_all, hid_dim=self.att_all, n_heads=2, dropout=0.1)
        self.attention_sample = MultiheadAttention(hid_dim1=self.att_all, hid_dim=self.att_all, n_heads=2, dropout=0.1)

    def forward(self, X1, X2, X3):
        encode1_1 = self.encoder1_1(X1)
        encode1_2 = self.encoder1_2(encode1_1)
        encode2_1 = self.encoder2_1(X2)
        encode2_2 = self.encoder2_2(encode2_1)
        encode3_1 = self.encoder3_1(X3)
        encode3_2 = self.encoder3_2(encode3_1)

        h1 = self.att_1(encode1_2)  # 维度转化为统一值
        h2 = self.att_2(encode2_2)
        h3 = self.att_3(encode3_2)

        # 各隐层拼接，统一维度
        enc_1 = torch.cat((encode1_1, encode1_2, h1), dim=1)
        enc_2 = torch.cat((encode2_1, encode2_2, h2), dim=1)
        enc_3 = torch.cat((encode3_1, encode3_2, h3), dim=1)
        fusion_1 = self.all_cat_B1(enc_1).unsqueeze(0)
        fusion_2 = self.all_cat_B2(enc_2).unsqueeze(0)
        fusion_3 = self.all_cat_B3(enc_3).unsqueeze(0)

        # attention
        Q_block = torch.cat((fusion_1, fusion_2, fusion_3), axis=0)
        Q_block = Q_block.permute(1, 0, 2)
        attn_block = self.attention_block_1(Q_block, Q_block, Q_block)
        # # 全部的静态信息
        out_static = torch.cat((attn_block, Q_block), dim=2)
        out_static = self.linear1(out_static)

        out_static = out_static.permute(1, 0, 2)

        # 计算样本间的残余信息
        attn_sample = self.attention_sample(out_static, out_static, out_static)

        # 补充块间的残余信息
        attn_sample = attn_sample.permute(1, 0, 2)
        Supplementary_blocks = torch.cat((attn_sample, Q_block), dim=2)
        Supplementary_blocks = self.linear1(Supplementary_blocks)
        Sup_attn_block = self.attention_block_2(Supplementary_blocks, Supplementary_blocks,
                                                Supplementary_blocks)


        out_fin =  torch.cat((Q_block, Sup_attn_block, attn_sample), dim=2)

        out_att_1 = out_fin[:, 0, :].squeeze(0)
        out_att_2 = out_fin[:, 1, :].squeeze(0)
        out_att_3 = out_fin[:, 2, :].squeeze(0)

        # 共同使用
        x_recon1 = self.decoder1_1(out_att_1)
        x_recon2 = self.decoder2_1(out_att_2)
        x_recon3 = self.decoder3_1(out_att_3)

        x_recon = torch.hstack((x_recon1, x_recon2, x_recon3))

        return x_recon, h1, h2, h3, x_recon1, x_recon2, x_recon3

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)

