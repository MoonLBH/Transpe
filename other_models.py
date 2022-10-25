from turtle import forward
import torch
import torch.nn as nn
from utils import *

class RNNpe(nn.Module):
    def __init__(self, d_model, hidden_dim, num_layers, hidden_dim2, dropout) -> None:
        '''
        hidden_dim: RNN神经元个数
        num_layers: RNN层数
        output_dim: 隐藏层输出维度
        '''
        super(RNNpe, self).__init__()

        self.d_model = d_model
        self.rna_emb = nn.Embedding(5, d_model)
        self.pos_emb = nn.Parameter(position_encoding(47, d_model), requires_grad=False)

        self.rnn = nn.RNN(d_model, hidden_dim, num_layers, nonlinearity='relu', bias=True, dropout=dropout)
        # self.fc = nn.Linear(hidden_dim, 1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim2, 1)
        )

    def forward(self, enc_input):
        rna_emb = self.rna_emb(enc_input[:,0,:].long())
        pos_emb = self.pos_emb
        seg_emb = nn.Parameter(torch.transpose(enc_input[:,1,:].unsqueeze(1).expand(len(enc_input),self.d_model,47),1,2), requires_grad=False)
        x = rna_emb + pos_emb + seg_emb
        x = x.transpose(0,1)
        out, h = self.rnn(x.float(), None)
        prediction = self.fc(out[-1])
        return prediction

class LSTMpe(nn.Module):
    def __init__(self, d_model, hidden_dim, num_layers, hidden_dim2, dropout) -> None:
        '''
        hidden_dim: lstm神经元个数
        num_layers: lstm层数
        output_dim: 隐藏层输出维度
        '''
        super(LSTMpe, self).__init__()

        self.d_model = d_model
        self.rna_emb = nn.Embedding(5, d_model)
        self.pos_emb = nn.Parameter(position_encoding(47, d_model), requires_grad=False)

        self.lstm = nn.LSTM(d_model, hidden_dim, num_layers, bias=True, dropout=dropout)
        # self.fc = nn.Linear(hidden_dim, 1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim2, 1)
        )

    def forward(self, enc_input):
        rna_emb = self.rna_emb(enc_input[:,0,:].long())
        pos_emb = self.pos_emb
        seg_emb = nn.Parameter(torch.transpose(enc_input[:,1,:].unsqueeze(1).expand(len(enc_input),self.d_model,47),1,2), requires_grad=False)
        x = rna_emb + pos_emb + seg_emb

        x = x.transpose(0,1)
        out, _ = self.lstm(x.float(), None)
        prediction = self.fc(out[-1])
        return prediction

class MLPpe(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout) -> None:
        super(MLPpe, self).__init__()

        self.d_model = d_model
        self.rna_emb = nn.Embedding(5, d_model)
        self.pos_emb = nn.Parameter(position_encoding(47, d_model), requires_grad=False)

        self.fc = nn.Sequential(
            nn.Flatten(),
            # nn.Linear(47*d_model, 47*d_model),
            # nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(47*d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, enc_input):
        rna_emb = self.rna_emb(enc_input[:,0,:].long())
        pos_emb = self.pos_emb
        seg_emb = nn.Parameter(torch.transpose(enc_input[:,1,:].unsqueeze(1).expand(len(enc_input),self.d_model,47),1,2), requires_grad=False)
        x = rna_emb + pos_emb + seg_emb
        out = self.fc(x)
        return out

class CNNpe(nn.Module):
    def __init__(self, d_model, conv_dim1, conv_dim2,  hidden_dim, dropout) -> None:
        super(CNNpe, self).__init__()
        self.d_model = d_model
        self.rna_emb = nn.Embedding(5, d_model)
        self.pos_emb = nn.Parameter(position_encoding(47, d_model), requires_grad=False)

        self.cnn_1 = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=conv_dim1, kernel_size = 4), # bs*d_model*47 -> bs*conv_dim1*44
            nn.ReLU(),
            nn.AvgPool1d(kernel_size = 2),  #bs*conv_dim1*44 -> bs*conv_dim1*22
            nn.Dropout(dropout)
            )
        self.cnn_2 = nn.Sequential(
            nn.Conv1d(in_channels=conv_dim1, out_channels=conv_dim2, kernel_size = 3), # bs*conv_dim1*22 -> bs*conv_dim2*20
            nn.ReLU(),
            nn.AvgPool1d(kernel_size = 2),  #bs*conv_dim2*44 -> bs*conv_dim2*10
            nn.Dropout(dropout)
            )
        # self.cnn_3 = nn.Sequential(
        #     nn.Conv1d(in_channels=conv_dim2, out_channels=conv_dim3, kernel_size = 3), # bs*conv_dim2*10 -> bs*conv_dim3*8
        #     nn.ReLU(),
        #     nn.AvgPool1d(kernel_size = 2),  #bs*conv_dim3*8 -> bs*conv_dim3*4
        #     nn.Dropout(dropout)
        #     )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(10*conv_dim2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )    

    def forward(self, enc_input):
        rna_emb = self.rna_emb(enc_input[:,0,:].long())
        pos_emb = self.pos_emb
        seg_emb = nn.Parameter(torch.transpose(enc_input[:,1,:].unsqueeze(1).expand(len(enc_input),self.d_model,47),1,2), requires_grad=False)
        x = rna_emb + pos_emb + seg_emb
        x = x.permute(0,2,1)
        x = self.cnn_1(x)
        x = self.cnn_2(x)
        # x = self.cnn_3(x)
        out = self.fc(x)
        return out



