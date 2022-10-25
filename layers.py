from turtle import forward
import torch
import torch.nn as nn
import numpy as np
from utils import *


class Attention(nn.Module):
    def __init__(self) -> None:
        super(Attention,self).__init__()
    
    def forward(self, Q, K, V, d_k, mask = None, dropout = None):
        scores = torch.matmul(Q,K.transpose(-1,-2))/ np.sqrt(d_k)

        if mask:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        scores = nn.Softmax(dim = -1)(scores)
        if dropout:
            scores = dropout(scores)
        
        output = torch.matmul(scores, V)
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,heads,dropout) -> None:
        super(MultiHeadAttention,self).__init__()

        self.d_model = d_model
        self.d_k = self.d_model // heads
        self.heads = heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, d_model)
        self.LN = nn.LayerNorm(d_model)

    def forward(self,q,k,v,mask = None):
        input, batch_size = q,q.size(0)
        # perform linear operation and split into N heads 
        Q = self.W_Q(q).view(batch_size, -1, self.heads, self.d_k)
        K = self.W_K(k).view(batch_size, -1, self.heads, self.d_k)
        V = self.W_V(v).view(batch_size, -1, self.heads, self.d_k)

        # transpose to get dimensions batch_size * heads * 47 * d_model'
        Q = Q.transpose(1,2)
        K = K.transpose(1,2)
        V = V.transpose(1,2)
        
        # calculate attention using function we will define next
        scores = Attention()(Q, K, V, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        scores = scores.transpose(1,2).contiguous().view(batch_size, -1 ,self.d_model)
        output = self.fc(scores)
        output = self.LN(input + output)
        return output
    
class FeedFoward(nn.Module):
    def __init__(self, d_model, d_ff, dropout) -> None:
        super(FeedFoward,self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.LN = nn.LayerNorm(d_model)
    
    def forward(self,inputs):
        input = inputs
        output = self.fc(inputs)
        output = self.LN(input + output)
        return output

class EncoderLayer(nn.Module):
    def __init__(self, heads, d_model, d_ff, dropout) -> None:
        super(EncoderLayer,self).__init__()

        self.encoder_attention = MultiHeadAttention(d_model, heads, dropout=dropout)
        self.feedfoward = FeedFoward(d_model,d_ff,dropout)

    def forward(self, enc_inputs, mask = None):
        output = self.encoder_attention(enc_inputs,enc_inputs,enc_inputs,mask)
        output = self.feedfoward(output)
        return output


class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, N, heads, dropout) -> None:
        super(Encoder, self).__init__()

        self.N = N
        self.d_model = d_model * heads
        self.rna_emb = nn.Embedding(5, d_model * heads)
        self.pos_emb = nn.Parameter(position_encoding(47, d_model * heads), requires_grad=False)

        self.layers = nn.ModuleList([EncoderLayer(heads, d_model * heads, d_ff, dropout) for _ in range(self.N)])
 
    def forward(self, enc_input, mask = None):
        rna_emb = self.rna_emb(enc_input[:,0,:].long())
        pos_emb = self.pos_emb
        seg_emb = nn.Parameter(torch.transpose(enc_input[:,1,:].unsqueeze(1).expand(len(enc_input),self.d_model,47),1,2), requires_grad=False)
        enc_output = rna_emb + pos_emb + seg_emb

        for layer in self.layers:
            enc_output = layer(enc_output, mask)        
        return enc_output
