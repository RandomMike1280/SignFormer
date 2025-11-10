import torch

n = 4
m = 4
x_values = torch.arange(1, n + 1).repeat_interleave(m)
y_values = torch.arange(1, m + 1).repeat(n)
pos = torch.stack((x_values, y_values), dim=1)
print(pos)

x = torch.tensor([[1,1], [1,2], [2,1], [2,2]])
e = torch.tensor([0.9, 0.8])

print(x*e)