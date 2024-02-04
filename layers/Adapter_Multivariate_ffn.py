import torch
import torch.nn as nn

class MyMappingNetwork_FFN(nn.Module):
    def __init__(self, input_dim, dropout):
        super(MyMappingNetwork_FFN, self).__init__()

        self.d_ff = int(input_dim / 2)
        # 定义7个线性层
        self.fc1 = nn.Linear(input_dim, self.d_ff)
        self.fc2 = nn.Linear(self.d_ff, input_dim)

        self.sigmoid = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(input_dim)

        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=input_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=input_dim, out_channels=input_dim, kernel_size=1)
        # self.dwqd =


    def forward(self, x, Multiple_i, Multivariate):


        ##################
        x_row = x
        x_fourier = self.fourier(x)
        x = x_fourier
        x = self.norm1(x + x_fourier)

        if Multiple_i == 0:
            x = self.dropout(self.sigmoid(self.fc1(x)))

            x_new = self.dropout(self.fc2(x))

            x_new = self.norm1(x_row + x_new)

        if Multiple_i == 0:
            x = x_new.reshape(-1, x_new.size(-2), x_new.size(-1)).transpose(-1, -2)
            x = self.dropout(self.sigmoid(self.conv1(x)))
            x = self.dropout(self.conv2(x)).transpose(-1, -2)
            x = x.reshape(-1, x_new.size(-3), x_new.size(-2), x_new.size(-1))

        x = x + x_new


        return self.norm1(x)

    def fourier(self, x):
        # shape of x: [B, L, D], B is batchsize, L is sequence length, D is hidden dimension

        # fft_hidden = torch.fft.fft(x, dim=-1)              # 沿着隐藏层维度进行FFT
        fft_seq = torch.fft.fft(x, dim=1)        # 沿着序列维度进行FFT
        out_put = torch.real(fft_seq)

        return out_put