from network import Network
from tokenizer import CharacterLevelTokenizer
import torch
import torch.nn.functional as F
import torch.optim as optim

tokenizer = CharacterLevelTokenizer.load("tokenizer.pkl")
model = Network(num_heads=6,
                num_layer=6,
                n_embed=384,
                context_window=512,
                vocab_size=len(tokenizer),
                dropout=0.2,
                device="cpu",
                dtype=torch.float32)

model.to(device="cpu", dtype=torch.float32)
model.train()

optimizer = optim.AdamW(model.parameters(), lr=3e-4)
criterion = torch.nn.CrossEntropyLoss()

with open("dataset.txt", encoding="utf-8") as f:
    text = f.read()

class Dataset(torch.utils.data.Dataset):
    def __init__(self, text, tokenizer, context_length):
        self.text = text
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.tokens = torch.tensor(tokenizer.encode(text))
        self.num_sequences = len(self.tokens) - context_length

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        l = len(self.tokens)
        x = self.tokens[idx:min(l, idx + self.context_length)]
        y = self.tokens[min(l, idx + 1):min(l, idx + self.context_length + 1)]
        return torch.tensor(x), torch.tensor(y)

context_window = 256
n = int(0.9 * len(text))
train_data = text[:n]
val_data = text[n:]
dataset = Dataset(train_data, tokenizer, context_length=context_window)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

for epoch in range(100):
    t = 0
    for x, y in dataloader: # iterating over batches
        optimizer.zero_grad()
        logits = model(x)
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        y = y.view(B*T)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        t += 1
        # print(f"Batch {t}/{len(dataloader)}")
        if t == 10:  # just to speed up training for testing
            break
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

#  = model.generate(torch.tensor([[0]]), max_length=64, temperature=1.0)

torch.save(model.state_dict(), "model.pt")