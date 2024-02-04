import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, PositionalEmbedding
import numpy as np
from model.RevIN import RevIN

from layers.Transformer_encoder_similarity import TransformerEncoder



class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.batch = configs.batch_size
        self.d_model = configs.d_model
        self.pred_len = configs.pred_len
        self.multivariate = configs.enc_in
        self.alpha = configs.alpha
        self.compare_baseline = configs.compare_baseline

        self.output_attention = configs.output_attention
        # Embedding
        # self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
        #                                             configs.dropout)

        self.PositionalEmbedding = PositionalEmbedding(configs.d_model)
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )


        ####### 扩维 ##
        self.mapping = nn.Parameter(torch.randn(configs.seq_len, configs.d_model))


        # self.embedding = nn.Linear(configs.seq_len, configs.d_model)

        self.dropout = nn.Dropout(0.1)


        # do patch
        # self.patch_len = 2
        # self.stride = 1
        # self.patch_num = int((configs.seq_len - self.patch_len) / self.stride + 1)
        # self.embedding_patch = nn.Linear(self.patch_len, configs.d_model)
        # self.seq_len = self.patch_num
        # self.seq_pred = nn.Linear(self.seq_len * configs.d_model, self.pred_len, bias=True)


        self.seq_pred = nn.Linear(self.seq_len * 16, configs.pred_len, bias=True)

        revin = True
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(self.multivariate, affine=True, subtract_last=False)

        self.flatten = nn.Flatten(start_dim=-2)


        self.LNorm = nn.LayerNorm(16)

        self.Linear_concat = nn.Linear(configs.d_model * 2, 16)

        # self.MyMappingNetwork_FFN = MyMappingNetwork_FFN(configs.d_model, configs.dropout)
        # similarity_heads = self.d_model // 8

        self.TransformerEncoder_similarity_up = TransformerEncoder(configs.e_layers, configs.d_model, configs.n_heads, self.seq_len, configs.d_ff, configs.dropout)

        # self.TransformerEncoder_similarity_up = TransformerEncoder(configs.e_layers, configs.d_model, configs.n_heads,
        #                                                         configs.d_ff, configs.dropout)


        # self.Flatten_Head = Flatten_Head(self.multivariate, configs.seq_len * configs.d_model * 2, configs.pred_len, head_dropout=0)



    def split_Multivariate(self, x, muti, x_enc):
        # x = 7 32 96 128     # 16 96 7
        x_enc = x_enc.permute(2, 0, 1)  # 7 16 96
        x_pooling = x[-1]  # 32 96 128
        x_pooling = torch.mean(x,dim=0)
        ############### OT 作为标准  #########
        x_enc_poling = x_enc[self.compare_baseline]  # 16 96

        alpha = self.batch * self.alpha

        # 计算所有余弦相似度
        cosine_similarities = F.cosine_similarity(x_enc_poling.unsqueeze(0), x_enc, dim=2)  # 1 16 96 和 7 16 96  # output 7 * 16
        cosine_similarities_sum = torch.sum(cosine_similarities, dim=1)  # sum over dim 2 -> 7

        # 根据阈值分类
        similarity_up_mask = cosine_similarities_sum >= alpha
        similarity_down_mask = cosine_similarities_sum < alpha

        similarity_up_id = torch.where(similarity_up_mask)[0]
        similarity_down_id = torch.where(similarity_down_mask)[0]

        similarity_up_all = x[similarity_up_mask].reshape(-1, self.seq_len, self.d_model)
        similarity_down_all = x[similarity_down_mask].reshape(-1, self.seq_len, self.d_model)

        if len(similarity_up_all) == 0:
            similarity_up_all = x_pooling
        if len(similarity_down_all) == 0:
            similarity_down_all = x_pooling

        return similarity_up_id, similarity_down_id, similarity_down_all, similarity_up_all

    def Embedding(self, x_enc):
        enc_out = x_enc.transpose(2, 1).unsqueeze(-1)

        enc_out = enc_out * self.dropout(self.mapping)            # 32 7 96 128
        ## 位置嵌入 ####
        enc_out_p = enc_out.reshape(-1, self.seq_len, self.d_model)
        enc_out_p = self.PositionalEmbedding(enc_out_p) + enc_out_p     # 224 96 128
        enc_out = enc_out_p.reshape(-1, self.multivariate, self.seq_len, self.d_model)  # 32 7 96 128

        enc_out_in = enc_out.transpose(1, 0)            # 7 32 96 128
        enc_out = enc_out_in

        return enc_out_in, enc_out

    def Channel_independence(self, enc_out_in, N):
        Multiple_i = 30
        enc_out_in = enc_out_in.reshape(-1, self.seq_len, self.d_model)
        enc_out_in = self.encoder(enc_out_in, Multiple_i, N, attn_mask=None)
        enc_out_in = enc_out_in.reshape(self.multivariate, -1, self.seq_len, self.d_model)  # 7 32 96 128

        return  enc_out_in

    def similarity_former(self, similarity_up_all, B):
        similarity_up_all = similarity_up_all.reshape(B, -1, self.seq_len, self.d_model).transpose(2, 1)        # batch 96 7 128
        similarity_up_all = similarity_up_all.reshape(B, similarity_up_all.size(2) * self.seq_len, self.d_model)

        enc_out_state = self.TransformerEncoder_similarity_up(similarity_up_all)    # 32 11 128
        enc_out_state = enc_out_state.reshape(B, self.seq_len, -1, self.d_model).permute(2, 0, 1, 3)

        return enc_out_state

    def unsimilarity_former(self, similarity_down_all, similarity_down_id, B, N):

        Multiple_i = 0      # 参数冻结
        # if len(similarity_down_id) >= 1:
        similarity_down_all = self.encoder(similarity_down_all, Multiple_i, N)
        similarity_down_all = similarity_down_all.reshape(-1, B, self.seq_len, self.d_model)

        return similarity_down_all

    def Time_Series_Restructuring(self, similarity_up_id, enc_out_state, similarity_down_all):

        enc_out_similarity = []
        k = 0
        j = 0
        for i in range(self.multivariate):
            if i in similarity_up_id:
                enc_out_similarity.append(enc_out_state[k])
                k = k + 1
            else:
                enc_out_similarity.append(similarity_down_all[j])
                j = j + 1
        enc_out_similarity = torch.stack(enc_out_similarity)

        return enc_out_similarity

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer


        x_enc = self.revin_layer(x_enc, 'norm')

        # x_enc_split = x_enc
        B, _, N = x_enc.shape  # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        # enc_out_multi_step = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens

        ##### do patch  ###
        # x_enc_patch = x_enc.transpose(2, 1).unfold(dimension=-1, size=self.patch_len, step=self.stride)  # 32 7 11 16
        # x_enc_patch = self.embedding_patch(x_enc_patch)
        # x_enc_patch = x_enc_patch.reshape(-1, self.patch_num, self.d_model)
        # x_enc_patch = self.PositionalEmbedding(x_enc_patch) + x_enc_patch
        # x_enc_patch = x_enc_patch.reshape(-1, N, self.patch_num, self.d_model)   # 16 7 11 128
        # enc_out_in = x_enc_patch.transpose(1, 0)
        # enc_out = enc_out_in
        # ######## 嵌入 ### 也可以考虑patch, 32 7 42 16, 然后16映射到128
        enc_out_in, enc_out = self.Embedding(x_enc)         # 7 32 96 128


        ################# 独立通道 ##########
        enc_out_in = self.Channel_independence(enc_out_in, N)

        ######### split_multivariate
        similarity_up_id, similarity_down_id, similarity_down_all, similarity_up_all = self.split_Multivariate(enc_out, self.multivariate, x_enc)


        ########### similarity ########### enc_out == unsimilarity || similarity
        enc_out_state = self.similarity_former(similarity_up_all, B)


        ########### unsimilarity ###########
        similarity_down_all = self.unsimilarity_former(similarity_down_all, similarity_down_id, B, N)



        ############## 序列组合  #############
        enc_out_similarity = self.Time_Series_Restructuring(similarity_up_id, enc_out_state, similarity_down_all)


        ######### concat 融合  ##############
        enc_out_concat = torch.cat((enc_out_in, enc_out_similarity), dim=-1)    # 7 32 96 256
        # enc_out_concat = torch.cat((enc_out_in * self.channel_independence_parameter, enc_out_similarity * self.Similarity_parameter), dim=-1)
        enc_out = self.Linear_concat(enc_out_concat)


        enc_out = self.LNorm(enc_out)

        #
        enc_out = enc_out.permute(1, 0, 2, 3)          # 32 7 96 128


        ##### enc_out_re = self.FC_1(enc_out)

        enc_out = self.flatten(enc_out)         # 32 7 96*16
        # enc_out = torch.relu(enc_out)
        dec_out = self.seq_pred(enc_out).permute(0, 2, 1)   # 32 192 7



        dec_out_ = self.revin_layer(dec_out, 'denorm')

        return dec_out_


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]    # [B, L, D]


class Flatten_Head(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):      # nf , 96*128
        super().__init__()

        # self.individual = individual
        self.n_vars = n_vars
        #
        # if self.individual:
        self.linears = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.flattens = nn.ModuleList()
        for i in range(self.n_vars):
            self.flattens.append(nn.Flatten(start_dim=-2))
            self.linears.append(nn.Linear(nf, target_window))
            self.dropouts.append(nn.Dropout(head_dropout))
        # else:
        #     self.flatten = nn.Flatten(start_dim=-2)
        #     self.linear = nn.Linear(nf, target_window)
        #     self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        # if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])  # z: [bs x d_model * patch_num]
                z = self.linears[i](z)  # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)  # x: [bs x nvars x target_window]
        # else:
        #     x = self.flatten(x)
        #     x = self.linear(x)
        #     x = self.dropout(x)
            return x