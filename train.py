from network import Network
from tokenizer import CharacterLevelTokenizer
import torch
import torch.nn.functional as F
import torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    from torch import amp
print(f"Using {device}")

tokenizer = CharacterLevelTokenizer.load("tokenizer.pkl")
model = Network(num_heads=4,
                num_layer=4,
                n_embed=384,
                context_window=512,
                vocab_size=len(tokenizer),
                dropout=0.2,
                device=device,
                dtype=torch.bfloat16)

model.to(device=device, dtype=torch.bfloat16)
model.train()

optimizer = optim.AdamW(model.parameters(), lr=3e-4)
criterion = torch.nn.CrossEntropyLoss()

class LazyDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, tokenizer, context_length, chunk_size=1000):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.chunk_size = chunk_size

        # Precompute file size
        with open(file_path, encoding="utf-8") as f:
            f.seek(0, 2)
            self.file_size = f.tell()

    def __len__(self):
        # Approximate: total size / chunk size
        return self.file_size // self.chunk_size

    def __getitem__(self, idx):
        with open(self.file_path, encoding="utf-8") as f:
            f.seek(idx * self.chunk_size)
            text_chunk = f.read(self.chunk_size)

        tokens = torch.tensor(self.tokenizer.encode(text_chunk), dtype=torch.long)

        # ensure enough tokens for context_length
        if len(tokens) < self.context_length + 1:
            raise IndexError("Not enough tokens in chunk.")

        start = torch.randint(0, len(tokens) - self.context_length - 1, (1,)).item()
        x = tokens[start:start+self.context_length]
        y = tokens[start+1:start+self.context_length+1]
        return x, y


# Usage
context_window = 512
dataset = LazyDataset("dataset.txt", tokenizer, context_length=context_window, chunk_size=5000)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

for epoch in range(1000):
    t = 0
    for x, y in dataloader:  # iterating over streamed batches
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        if device == "cuda":
            with amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x)
                    B, T, C = logits.shape
                    logits = logits.view(B*T, C)
                    y = y.view(B*T)
                    loss = criterion(logits, y)
        else:
            logits = model(x)
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            y = y.view(B*T)
            loss = criterion(logits, y) 
        loss.backward()
        optimizer.step()
        t += 1
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

#  = model.generate(torch.tensor([[0]]), max_length=64, temperature=1.0)

torch.save(model.state_dict(), "model.pt")