import torch
import torch.nn as nn
import math

class PosEmbedding(nn.Module):
    def __init__(self, input_dim, context_window, device="cpu", dtype=torch.float16):
        super(PosEmbedding, self).__init__()
        self.pos = torch.arange(0, context_window, dtype=dtype).unsqueeze(1)
        self.dim_idx = torch.arange(0, input_dim, 2, dtype=dtype)
        self.to(device=device, dtype=dtype)
        self.context_window = context_window
        self.input_dim = input_dim
        self.device = device
        self.dtype = dtype
        self.to(device=device, dtype=dtype)

        pe = torch.zeros(self.context_window, self.input_dim, dtype=self.dtype, device=self.device)
        div_term = torch.exp(self.dim_idx * (-math.log(10000.0) / self.input_dim))
        pe[:, 0::2] = torch.sin(self.pos * div_term)
        pe[:, 1::2] = torch.cos(self.pos * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x

PosEmbedding = PosEmbedding(input_dim=1024, context_window=1024, device="cpu", dtype=torch.float16)
x = torch.randn(1, 32, 1024)
y = PosEmbedding(x)
print(y.shape)