import torch
import json
import csv
from tqdm import tqdm
from pprint import pprint
import numpy as np
from torch import nn, optim

def apply_mask(x, len_list):
        for i in range(x.shape[0]):
            x[i, len_list[i]:, :] = 0
        return x

class PosEncoding(nn.Module):
    def __init__(self):
        super(PosEncoding, self).__init__()
        self.max_len = 1000
        self.embedding_dim = 400
        #self.embedding_layer = nn.Embedding(len(word2index), self.embedding_dim)
    def forward(self, embedding):
        # embeddings (b, s, d)
        # pos runs from 0, s-1
        # i runs from 0, d-1
        #embedding = self.embedding_layer(x)
        pos_max = embedding.shape[1]
        d_max = embedding.shape[2]
        # create pos encoding matrix of size (b, s, d)
        pos_encoding = torch.zeros((pos_max, d_max))
        for pos in range(pos_max):
            for i in range(0, d_max, 2):
                pos_encoding[pos, i] = np.sin(pos / (10000 ** ((2 * i)/d_max)))
                pos_encoding[pos, i+1] = np.cos(pos / (10000 ** ((2 * (i+1))/d_max)))
        pos_encoding = pos_encoding.unsqueeze(0)
        pos_encoding = pos_encoding.repeat(embedding.shape[0], 1, 1)
        #print(embedding.shape, pos_encoding.shape)
        embedding = embedding + pos_encoding.to(embedding.device)
        return embedding
    




class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, dim):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        # QKV for multihead attention
        self.Q = nn.Linear(self.dim, self.dim)
        self.K = nn.Linear(self.dim, self.dim)
        self.V = nn.Linear(self.dim, self.dim)
        self.softmax = nn.Softmax(dim=-1)
        self.layer_norm = nn.LayerNorm(self.dim)
    def forward(self, embedded):
        # implement multi head attention
        q_val = self.Q(embedded)
        k_val = self.K(embedded)
        v_val = self.V(embedded)
        # print q_val, k_val, v_val completely
        # print(q_val)
        # print(k_val)
        # print(v_val)
        #print(q_val.shape, k_val.shape, v_val.shape)
        # reshape q, k, v to (batch_size, num_heads, seq_len, dim/num_heads)
        q_val = q_val.view(q_val.shape[0], q_val.shape[1], self.num_heads, q_val.shape[2]//self.num_heads)
        k_val = k_val.view(k_val.shape[0], k_val.shape[1], self.num_heads, k_val.shape[2]//self.num_heads)
        v_val = v_val.view(v_val.shape[0], v_val.shape[1], self.num_heads, v_val.shape[2]//self.num_heads)
        #print(q_val.shape, k_val.shape, v_val.shape)
        # transpose to (batch_size, num_heads, dim/num_heads, seq_len)
        q_val = q_val.transpose(1, 2)
        k_val = k_val.transpose(1, 2)
        v_val = v_val.transpose(1, 2)
        #print(q_val.shape, k_val.shape, v_val.shape)  
        # multiply q and k transpose
        qk_val = torch.matmul(q_val, k_val.transpose(-2, -1))
        qk_val[qk_val != qk_val] = float('-inf')
        # scale qk_val
        qk_val = qk_val / np.sqrt(qk_val.shape[-1])
        # apply softmax
        #qk_val = self.softmax(qk_val)
        # apply softmax such that all -infs and nans go to 0
        qk_val1 = torch.softmax(qk_val, dim=-1)
        qk_val = qk_val1.clone()
        nan_mask = torch.isnan(qk_val)
        # Replace NaNs with 0
        qk_val[nan_mask] = 0
        v_val[v_val != v_val] = 0
        #print("----------__>", qk_val.shape, v_val.shape)
        z_init = torch.matmul(qk_val, v_val)
        #print(z_init.shape)
        #use tensor.view(tensor.size(0), tensor.size(2), -1)
        z = z_init.view(z_init.shape[0], z_init.shape[2], -1)
        # add and norm
        z = self.layer_norm(embedded + z)
        return z
class FFN(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(input_dim, 1024)
        self.linear2 = nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(output_dim)
        # dropout
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        y = self.linear1(x)
        y = self.relu(y)
        y = self.linear2(y)
        y = self.dropout(y)
        y = self.layer_norm(y+x)
        return y
    
class EncodingLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super(EncodingLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.multi_head_attention = MultiHeadAttention(num_heads, d_model)
        self.ffn = FFN(d_model, d_model, dropout)
    def forward(self, x):
        # first raise error id d_model is not divisible by num_heads
        if x.shape[-1] % self.multi_head_attention.num_heads != 0 or self.d_model % self.multi_head_attention.num_heads != 0:
            raise ValueError("Embedding Dimension is not divisible by Number of Heads")
        # apply multi head attention
        attn_output = self.multi_head_attention(x)
        # apply ffn
        ffn_output = self.ffn(attn_output)
        return ffn_output


