import torch
import torch.nn as nn
import math


class InputEmbedding(nn.Module):
    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.d_model)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, seq_len:int, dropout:float)->None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on seq len and d model
        
        pe=torch.zeros(seq_len, d_model)
        position=torch.arange(0, seq_len).unsqueeze(1) # (seq_len, 1)
        div_term=torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) # even indices
        pe[:, 1::2] = torch.cos(position * div_term) # odd indices
        
        pe = pe.unsqueeze(0) # add batch dimension (1, seq_len, d_model)
        self.register_buffer('pe', pe) # register the buffer to save it in the model's state_dict
        
        def forward(self, x):
            x = x + (self.pe[:, :x.size(1), :].requires_grad_(False))
            return self.dropout(x)
        

    