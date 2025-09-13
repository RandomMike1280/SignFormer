import torch
from torch._dynamo.backends.common import dtype_from_inputs
import torch.nn as nn
import torch.nn.functional as F
import math
from tokenizer import CharacterLevelTokenizer

class dot_product_attention(nn.Module):
    def __init__(self, dropout=0.1, device="cpu", dtype=torch.bfloat16, input_dim=1024, head_size=32, context_window=1024):
        super(dot_product_attention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.dtype = dtype
        self.head_size = head_size

        try:
            mask = torch.tril(torch.ones((context_window, context_window), device=device, dtype=dtype))
            self.register_buffer("mask", mask)
        except RuntimeError:
            mask = torch.tril(torch.ones((context_window, context_window), device=device, dtype=torch.bfloat16))
            self.register_buffer("mask", mask) 

        self.qkv = nn.Linear(input_dim, 3 * head_size, dtype=dtype, bias=False)
        self.to(device=device, dtype=dtype)

    def forward(self, x):
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1) # (B, T, head_size) each
        k_t = k.transpose(-2, -1) # (B, C, T)
        B, T, C = x.shape


        dots = torch.matmul(q, k_t) * (self.head_size ** -0.5) # (B, T, T)
        dots = dots.masked_fill(self.mask[:T, :T] == 0, float('-inf')) # (B, T, T)
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v) # (B, T, head_size)
        return out

class Multihead_Attention(nn.Module):
    def __init__(self, num_heads, n_embed, context_window, head_size, dropout=0.1, device="cpu", dtype=torch.bfloat16):
        super(Multihead_Attention, self).__init__()
        self.heads = nn.ModuleList()
        for _ in range(num_heads):
            self.heads.append(dot_product_attention(dropout=dropout, device=device, dtype=dtype, input_dim=n_embed, context_window=context_window, head_size=head_size))
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
        self.to(device=device, dtype=dtype)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, input_dim, dropout=0.1, device="cpu", dtype=torch.bfloat16):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim*4, dtype=dtype)
        self.fc2 = nn.Linear(input_dim*4, input_dim, dtype=dtype)
        self.dropout = nn.Dropout(dropout)
        self.to(device=device, dtype=dtype)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout((self.fc2(x)))
        return x

class TransformerBlock(nn.Module):
    def __init__(self, num_heads, n_embed, context_window, dropout=0.1, device="cpu", dtype=torch.bfloat16):
        super(TransformerBlock, self).__init__()
        head_size = n_embed // num_heads
        self.attn = Multihead_Attention(num_heads, n_embed, context_window, head_size=head_size, dropout=dropout, device=device, dtype=dtype)
        self.ff = FeedForward(n_embed, dropout=dropout, device=device, dtype=dtype)
        self.ln1 = nn.LayerNorm(n_embed, dtype=dtype)
        self.ln2 = nn.LayerNorm(n_embed, dtype=dtype)
        self.to(device=device, dtype=dtype)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class PosEmbedding(nn.Module):
    def __init__(self, input_dim, context_window, device="cpu", dtype=torch.bfloat16):
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

class Network(nn.Module):
    def __init__(self, num_heads, num_layer, n_embed, context_window, vocab_size, dropout=0.1, device="cpu", dtype=torch.bfloat16):
        super(Network, self).__init__()
        self.emb = nn.Embedding(vocab_size, n_embed, dtype=dtype)
        # self.pos_embed = PosEmbedding(input_dim, context_window, device=device, dtype=dtype)
        self.pos_embed = nn.Embedding(context_window, n_embed, dtype=dtype)
        self.transformer = nn.ModuleList()
        for _ in range(num_layer):
            self.transformer.append(TransformerBlock(num_heads, n_embed, context_window, dropout=dropout, device=device, dtype=dtype))
        self.transformer.append(nn.LayerNorm(n_embed, dtype=dtype))
        self.out = nn.Linear(n_embed, vocab_size, dtype=dtype)
        # self.emb.weight = self.out.weight
        self.to(device=device, dtype=dtype)
        self.dtype = dtype
        self.device = device
        self.d_model = n_embed
        self.context_window = context_window

    def forward(self, x):
        B, T = x.shape
        idx = torch.arange(T, device=self.device)
        x = x.to(device=self.device, dtype=torch.int32)
        x = self.emb(x) * (self.d_model ** 0.5)
        x += self.pos_embed(idx)
        for block in self.transformer:
            x = block(x)
        # x = self.out(x[:, -1, :])
        x = self.out(x)
        return x

    def generate(self, x, max_length:int, temperature:float, stop_token:str=None):
        if len(x.size()) == 1:
            x = x.unsqueeze(0)
        self.eval()
        if stop_token:
            while x.size(1) < self.context_window:
                probs = self.forward(x)
                probs = probs[:, -1, :]
                probs = F.softmax(probs / temperature, dim=-1)
                token = torch.multinomial(probs, 1)
                x = torch.cat([x, token], dim=-1)
                if x == stop_token:
                    break
        else:
            with torch.no_grad():
                for _ in range(max_length):
                    probs = self.forward(x)
                    probs = probs[:, -1, :]
                    # print(probs.shape)
                    probs = F.softmax(probs / temperature, dim=-1)
                    token = torch.multinomial(probs, 1)
                    x = torch.cat([x, token], dim=-1)
        return x
        

if __name__ == "__main__":
    from summary import ModelSummary
    with open("dataset.txt", "r") as f:
        text = f.read()
    tokenizer = CharacterLevelTokenizer().load("tokenizer.pkl")
    model = Network(num_heads=4,
                num_layer=4,
                n_embed=128,
                context_window=512,
                vocab_size=65,
                dropout=0.1,
                device="cpu",
                dtype=torch.float32)
    model.load_state_dict(torch.load("model.pt"))
    ModelSummary(model)
    x = torch.tensor(tokenizer.encode("C")).unsqueeze(0)
    # print(x)
    y = model(x)
    # print(y.shape)
    seq = model.generate(x, max_length=500, temperature=1.0)
    # print(seq)
    seq = list(seq[0].tolist())
    print(seq)
    print(tokenizer.decode(seq))