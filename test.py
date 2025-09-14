import torch
from torchtune.modules import KVCache

cache = KVCache(batch_size=2, max_seq_len=16, num_kv_heads=1, head_dim=32, dtype=torch.bfloat16)
keys, values = torch.ones((2, 1, 8, 32), dtype=torch.bfloat16), torch.ones((2, 1, 8, 32), dtype=torch.bfloat16)
cache.update(keys, values)
# now positions 0 through 7 are filled
print(cache.size)

keys, values = torch.ones((2, 1, 1, 32), dtype=torch.bfloat16), torch.ones((2, 1, 1, 32), dtype=torch.bfloat16)
k, v = cache.update(keys, values)
print(k.shape, v.shape)  # should be (2, 1, 16, 32)