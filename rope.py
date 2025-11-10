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
        emb = freqs.repeat_interleave(2, dim=-1)
        cos_vals, sin_vals = emb.cos(), emb.sin()

        self.register_buffer("cos_vals", cos_vals)
        self.register_buffer("sin_vals", sin_vals)

    def test(self):
        d_model = self.d_model
        max_context_window = self.max_context_window
        dtype = self.dtype
        dim_idx = torch.arange(0, d_model, 2, dtype=dtype)
        div_term = torch.exp(dim_idx * (-math.log(10000.0) / d_model))
        pos = torch.arange(max_context_window, dtype=dtype)
        freqs = torch.einsum("i,j->ij", pos, div_term)
        emb = torch.cat((freqs, freqs), dim=-1)
        print(emb)
        print(self.cos_vals.shape)
        print(self.sin_vals.shape)
        print(self.cos_vals)

    def forward(self, x):
        B, T, C = x.shape
        cos = self.cos_vals[:T, :]
        sin = self.sin_vals[:T, :]
        x_rotated = x * cos.unsqueeze(0) + interleave(x) * sin.unsqueeze(0)

        return x_rotated

class RoPE2D(nn.Module):
    def __init__(self, d_model, max_context_window, device="cpu", dtype=torch.bfloat16):
        super(RoPE2D, self).__init__()
        self.device = device
        self.dtype = dtype
        self.max_context_window = max_context_window
        self.d_model = d_model
        self.to(device=device, dtype=dtype)
        dim = int(max_context_window**0.5)

        dim_idx = torch.arange(0, d_model, 2, dtype=dtype)
        div_term = torch.exp(dim_idx * (-math.log(10000.0) / d_model))
        x_values = torch.arange(1, dim + 1).repeat_interleave(dim)
        y_values = torch.arange(1, dim + 1).repeat(dim)
        pos_xy = torch.stack((x_values, y_values), dim=1)
        freqs_x = torch.einsum("i,j->ij", pos_xy[:, 0], div_term[..., 0::2])
        freqs_y = torch.einsum("i,j->ij", pos_xy[:, 1], div_term[..., 1::2])
        emb = interleave_concat(freqs_x, freqs_y)
        cos_vals, sin_vals = emb.cos(), emb.sin()

        self.register_buffer("cos_vals", cos_vals)
        self.register_buffer("sin_vals", sin_vals)

    def test(self):
        d_model = self.d_model
        max_context_window = self.max_context_window
        dim = int(max_context_window**0.5)
        dtype = self.dtype
        dim_idx = torch.arange(0, d_model, 2, dtype=dtype)
        div_term = torch.exp(dim_idx * (-math.log(10000.0) / d_model))
        x_values = torch.arange(1, dim + 1).repeat_interleave(dim)
        y_values = torch.arange(1, dim + 1).repeat(dim)
        pos_xy = torch.stack((x_values, y_values), dim=1)
        freqs_x = torch.einsum("i,j->ij", pos_xy[:, 0], div_term[..., 0::2])
        freqs_y = torch.einsum("i,j->ij", pos_xy[:, 1], div_term[..., 1::2])
        emb = interleave_concat(freqs_x, freqs_y).repeat_interleave(2, dim=-1)
        cos_vals, sin_vals = emb.cos(), emb.sin()
        print(emb)
        print(emb.shape)

    def forward(self, x):
        B, T, C = x.shape
        cos = self.cos_vals[:T, :]
        sin = self.sin_vals[:T, :]
        x_rotated = x * cos.unsqueeze(0) + interleave(x) * sin.unsqueeze(0)

        return x_rotated

def interleave_concat(x1, x2):
    T, C = x1.shape
    x1_reshaped = x1.reshape(T*C, 1)
    x2_reshaped = x2.reshape(T*C, 1)
    x_rot_half = torch.cat((x1_reshaped, x2_reshaped), dim=-1).reshape(T, C*2)
    return x_rot_half

def interleave(x, negative=True):
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
    rope = RoPE2D(d_model=8, max_context_window=16)
    rope.test()

    dummy_q = torch.rand(1, 4, 4)
    # rot_q = rope(dummy_q)
    # print(rot_q.shape)