from network import Network
from tokenizer import CharacterLevelTokenizer
import torch
import torch.nn.functional as F
import torch.optim as optim
import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    from torch import amp
print(f"Using {device}")

tokenizer = CharacterLevelTokenizer("AĂÂBCDĐEÊGHIKLMNOÔƠPQRSTUƯVXYaăâbcdđeêghiklmnnoôơpqrstuưvxyÁÀẢÃẠẮẰẲẴẶẤẦẨẪẬÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ~`!@#$%^&*()_+-=[]{}\\|;:'\",.<>/? 0123456789\n")
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

from datasets import load_dataset

class HuggingFaceDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, context_length):
        # Load the Vietnamese poetry dataset
        self.dataset = load_dataset("bigscience-data/roots_vi_vietnamese_poetry")["train"]
        self.tokenizer = tokenizer
        self.context_length = context_length
        
        # Preprocess all texts and concatenate them
        self.all_tokens = []
        for item in self.dataset:
            tokens = self.tokenizer.encode(item["text"])
            self.all_tokens.extend(tokens)
        self.all_tokens = torch.tensor(self.all_tokens, dtype=torch.long)

    def __len__(self):
        return max(0, len(self.all_tokens) - self.context_length - 1)

    def __getitem__(self, idx):
        x = self.all_tokens[idx:idx + self.context_length]
        y = self.all_tokens[idx + 1:idx + self.context_length + 1]
        return x, y


# Usage
context_window = 512
dataset = HuggingFaceDataset(tokenizer, context_length=context_window)
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