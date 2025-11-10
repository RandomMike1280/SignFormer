import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RoPE1D(nn.Module):
    def __init__(self, d_model, max_context_window, device="cpu", dtype=torch.bfloat16):
        super(RoPE1D, self).__init__()
        self.device = device
        self.dtype = dtype
        self.max_context_window = max_context_window
        self.d_model = d_model
        self.to(device=device, dtype=dtype)

        dim_idx = torch.arange(0, d_model, 2, dtype=dtype)
        div_term = torch.exp(dim_idx * (-math.log(10000.0) / d_model))
        pos = torch.arange(max_context_window, dtype=dtype)
        freqs = torch.einsum("i,j->ij", pos, div_term)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_vals, sin_vals = emb.cos(), emb.sin()

        self.register_buffer("cos_vals", cos_vals)
        self.register_buffer("sin_vals", sin_vals)

    def test(self):
        print(self.cos_vals.shape)
        print(self.sin_vals.shape)
        print(self.cos_vals)

    def forward(self, x):
        B, T, C = x.shape
        cos = self.cos_vals[:T, :]
        sin = self.sin_vals[:T, :]
        x_rotated = x * cos.unsqueeze(0) + interveave(x) * sin.unsqueeze(0)

        return x_rotated

class RoPE2D(nn.Module):
    def __init__(self, d_model, max_context_window, device="cpu", dtype=torch.bfloat16):
        super(RoPE2D, self).__init__()
        self.device = device
        self.dtype = dtype
        self.max_context_window = max_context_window
        self.d_model = d_model
        self.to(device=device, dtype=dtype)

        dim_idx = torch.arange(0, d_model, 2, dtype=dtype)
        div_term = torch.exp(dim_idx * (-math.log(10000.0) / d_model))
        pos_x = torch.arange(max_context_window, dtype=dtype)
        pos_y = torch.arange(max_context_window, dtype=dtype)
        freqs_x = torch.einsum("i,j->ij", pos_x, div_term)
        freqs_y = torch.einsum("i,j->ij", pos_y, div_term)
        emb = torch.cat((freqs_x, freqs_y), dim=-1)
        cos_vals, sin_vals = emb.cos(), emb.sin()

        self.register_buffer("cos_vals", cos_vals)
        self.register_buffer("sin_vals", sin_vals)

    def test(self):
        print(self.cos_vals.shape)
        print(self.sin_vals.shape)

    def forward(self, x):
        B, T, C = x.shape
        cos = self.cos_vals[:T, :]
        sin = self.sin_vals[:T, :]
        x_rotated = x * cos.unsqueeze(0) + interveave(x) * sin.unsqueeze(0)

        return x_rotated

def interveave(x, negative=True):
    B, T, C = x.shape
    x_reshaped = x.reshape(B, T, C//2, 2)
    x1 = x_reshaped[..., 0::2]
    x2 = x_reshaped[..., 1::2]
    if negative:
        x_rot_half = torch.cat((-x2, x1), dim=-1).reshape(B, T, C)
    else:
        x_rot_half = torch.cat((x2, x1), dim=-1).reshape(B, T, C)
    return x_rot_half

if __name__ == "__main__":
    rope = RoPE1D(d_model=4, max_context_window=8)
    rope.test()

    dummy_q = torch.rand(1, 8, 4)
    rot_q = rope(dummy_q)
    # print(rot_q.shape)