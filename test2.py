import torch

def interleave_concat(x1, x2):
    T, C = x1.shape
    x1_reshaped = x1.flatten().unsqueeze(-1).transpose(-2, -1)
    x2_reshaped = x2.flatten().unsqueeze(-1).transpose(-2, -1)
    x_rot_half = torch.cat((x1_reshaped, x2_reshaped), dim=-1).reshape(T, C*2)
    return x_rot_half

x = torch.randn(2, 4)
y = torch.randn(2, 4)
print(x)
print(y)
print(interleave_concat(x, y))
