import torch
import torch.nn as nn
import torch
import torch.nn.functional as F


class SelfAttention(torch.nn.Module):
    def __init__(self, embed_size, heads, seq_len, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = self.embed_size // heads
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)


        self.setting_size = 16
        #######
        self.fc_1q = torch.nn.Linear(self.embed_size, self.setting_size)
        self.fc_2q = torch.nn.Linear(self.seq_len, self.seq_len // 4)  # 96 / 4 = 24
        self.fc_3q = torch.nn.Linear((self.seq_len // 4) * self.setting_size, self.embed_size)

        # self.fc_1k = torch.nn.Linear(self.embed_size, self.setting_size)
        # self.fc_2k = torch.nn.Linear(self.seq_len, self.seq_len // 4)  # 96 / 4 = 24
        self.fc_3k = torch.nn.Linear((self.seq_len // 4) * self.setting_size, self.embed_size)

        # self.fc_1v = torch.nn.Linear(self.embed_size, self.setting_size)
        # self.fc_2v = torch.nn.Linear(self.seq_len, self.seq_len // 4)  # 96 / 4 = 24
        self.fc_3v = torch.nn.Linear((self.seq_len // 4) * self.setting_size, self.embed_size)

        self.fc_out_1 = torch.nn.Linear(self.embed_size, (self.seq_len // 4) * self.setting_size)
        self.fc_out_2 = torch.nn.Linear(self.seq_len // 4, self.seq_len)
        self.fc_out_3 = torch.nn.Linear(self.setting_size, self.embed_size)

        # self.activate = nn.Sigmoid()
        self.norm1 = nn.LayerNorm((self.seq_len // 4) * self.setting_size)
        self.norm2 = nn.LayerNorm((self.seq_len // 4) * self.setting_size)


    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]


        multivariate = query_len // self.seq_len


        ####################
        # multivariate = query_len // self.seq_len
        if multivariate > 1:
            query = self.dropout(self.fc_1q(query))
            # query = query.reshape(N, multivariate, self.seq_len, self.setting_size).transpose(3, 2)
            query = query.reshape(N, self.seq_len, multivariate, self.setting_size).transpose(2, 1).transpose(3, 2)
            query = self.dropout( self.fc_2q(query) )
            query = keys = values = self.norm1(query.reshape(N, multivariate, -1))
            query = self.dropout(self.fc_3q(query))

            # keys = self.dropout(self.fc_1k(keys))
            # keys = keys.reshape(N, multivariate, self.seq_len, 16).transpose(3, 2)
            # keys = self.dropout(self.fc_2k(keys))
            # keys = keys.reshape(N, multivariate, -1)
            keys = self.dropout(self.fc_3k(keys))

            # values = self.fc_1v(values)
            # values = values.reshape(N, multivariate, self.seq_len, 16).transpose(3, 2)
            # values = self.dropout(self.fc_2v(values))
            # values = values.reshape(N, multivariate, -1)
            values = self.dropout(self.fc_3v(values))
            #########


            values = values.reshape(N, multivariate, self.heads, self.head_dim)
            keys = keys.reshape(N, multivariate, self.heads, self.head_dim)
            queries = query.reshape(N, multivariate, self.heads, self.head_dim)



            energy_out = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

            if mask is not None:
                energy_out = energy_out.masked_fill(mask == 0, float("-1e20"))  # Apply mask

            attention = torch.nn.functional.softmax(energy_out / (self.embed_size ** (1 / 2)), dim=3)  # batch * head * len * len
            # attention = energy_out

            out_sim = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, multivariate, self.heads * self.head_dim)   # 32 multivariate heads head_dim

            out_sim = self.dropout( self.norm2(self.fc_out_1(out_sim)) )
            out_sim = out_sim.reshape(N, multivariate, self.setting_size, self.seq_len // 4)
            out_sim = self.dropout( self.fc_out_2(out_sim) ).transpose(3, 2)
            out_sim = self.dropout(self.fc_out_3(out_sim))  # 32 multi 96 128
            out = out_sim.reshape(N, multivariate * self.seq_len, self.embed_size)

        else:
            out = query


        return out


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, seq_len, dim_feedforward=128, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn = SelfAttention(d_model, nhead, seq_len, dropout=dropout)
        self.feedforward = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=dim_feedforward, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=dim_feedforward, out_channels=d_model, kernel_size=1)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, mask=None):
        src2 = self.self_attn(src, src, src, mask)
        src = src + self.dropout(src2)
        src = self.norm1(src)

        src2 = self.feedforward(src.transpose(-1, 1)).transpose(-1, 1)
        src = src + self.dropout(src2)
        src = self.norm2(src)

        return src

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, seq_len, dim_feedforward=128, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, nhead, seq_len, dim_feedforward, dropout) for _ in range(num_layers)])

    def forward(self, src):
        for layer in self.layers:
            src = layer(src)
        return src
