import torch

logvar = torch.randn(1, 4, 4, dtype=torch.bfloat16)
std = torch.exp(0.5 * logvar)
eps = torch.randn_like(std)
mu = torch.randn(1, 4, 4, dtype=torch.bfloat16)
z = mu + std * eps

# std = e^logvar/2
# eps = N(0,1)
# z = mu + std * eps 

# print(logvar)
print(std)
print(eps)