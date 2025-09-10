from network import Network
from tokenizer import CharacterLevelTokenizer
import torch
import torch.nn.functional as F
import torch.optim as optim

tokenizer = CharacterLevelTokenizer.load("tokenizer.pkl")
model = Network(num_heads=8,
                num_layer=6,
                input_dim=1024,
                context_window=1024,
                dim=256,
                hidden_dim=256,
                vocab_size=len(tokenizer),
                dropout=0.1,
                device="cpu",
                dtype=torch.float32)

model.to(device="cpu", dtype=torch.float32)
model.train()

optimizer = optim.Adam(model.parameters(), lr=0.001)
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
        seq = self.tokens[idx:idx + self.context_length]
        target = self.tokens[idx + self.context_length]
        return torch.tensor(seq), torch.tensor(target)

dataset = Dataset(text, tokenizer, context_length=64)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

for epoch in range(10):
    t = 0
    for sequences, targets in dataloader:
        optimizer.zero_grad()
        logits = model(sequences)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        t += 1
        print(f"Batch {t}/{len(dataloader)}")
        if t == 100:
            break
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

#  = model.generate(torch.tensor([[0]]), max_length=64, temperature=1.0)

torch.save(model.state_dict(), "model.pt")