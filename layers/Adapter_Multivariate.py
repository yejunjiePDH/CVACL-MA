import torch
import torch.nn as nn

class MyMappingNetwork(nn.Module):
    def __init__(self, input_dim, dropout):
        super(MyMappingNetwork, self).__init__()

        self.d_ff = int(input_dim / 2)
        # 定义7个线性层
        self.fc1 = nn.Linear(input_dim, self.d_ff)
        self.fc2 = nn.Linear(self.d_ff, input_dim)

        self.fc3 = nn.Linear(input_dim, self.d_ff)
        self.fc4 = nn.Linear(self.d_ff, input_dim)

        self.relu = nn.GELU()

        self.sigmoid = nn.Sigmoid()

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(input_dim)



    def forward(self, x, Multiple_i, Multivariate):


        ##################
        if Multiple_i == 0:
            x = self.dropout(self.relu(self.fc1(x)))
            # x = self.relu(x)
            x = self.dropout(self.fc2(x))

        if Multiple_i == 1:
            x = self.dropout(self.relu(self.fc3(x)))
            # x = self.relu(x)
            x = self.dropout(self.fc4(x))


        return self.norm1(x)
