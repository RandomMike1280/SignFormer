import torch
from torch._dynamo.backends.common import dtype_from_inputs
import torch.nn as nn
import torch.nn.functional as F
import math
from tokenizer import CharacterLevelTokenizer
from torchtune.modules import KVCache

class dot_product_attention(nn.Module):
    def __init__(self, dropout=0.1, use_kv_cache=False, device="cpu", dtype=torch.bfloat16, input_dim=1024, head_size=32, context_window=1024, use_mask=True):
        super(dot_product_attention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.dtype = dtype
        self.head_size = head_size
        self.context_window = context_window
        if use_mask:
            try:
                mask = torch.tril(torch.ones((context_window, context_window), device=device, dtype=dtype))
                self.register_buffer("mask", mask)
            except RuntimeError:
                mask = torch.tril(torch.ones((context_window, context_window), device=device, dtype=torch.bfloat16))
                self.register_buffer("mask", mask) 
        self.use_mask = use_mask
        self.qkv = nn.Linear(input_dim, 3 * head_size, dtype=dtype, bias=False)
        self.to(device=device, dtype=dtype)
        self.use_kv_cache = use_kv_cache
        self.kv_cache = None

    def forward(self, x):
        B, T, C = x.shape
        
        if self.use_kv_cache:
            if self.kv_cache is None:
                qkv = self.qkv(x)
                q, k, v = qkv.chunk(3, dim=-1) # (B, T, head_size) each
                k_t = k.transpose(-2, -1) # (B, C, T) or k.view(B, C, T)
                self.kv_cache = KVCache(batch_size=B,
                                        max_seq_len=self.context_window,
                                        num_kv_heads=1,
                                        head_dim=self.head_size,
                                        dtype=self.dtype
                                        )
                
                self.kv_cache = self.kv_cache.to(self.device)
                kf, vf = k.unsqueeze(1), v.unsqueeze(1) # (B, 1, T, head_size) for the num heads dimension

                cached_k, cached_v =self.kv_cache.update(kf, vf)
            else:
                last_token = x[:, -1:, :] # (B, 1, C)
                qkv = self.qkv(last_token)
                q, k, v = qkv.chunk(3, dim=-1) # (B, 1, head_size) each

                kf, vf = k.unsqueeze(1), v.unsqueeze(1) # (B, 1, 1, head_size) for the num heads dimension
                cached_k, cached_v = self.kv_cache.update(kf, vf)
            cached_k, v = cached_k.squeeze(1), cached_v.squeeze(1) # (B, T, head_size)
            cached_k, v = cached_k[:, :T, :], v[:, :T, :] # (B, T, head_size)
            k_t = cached_k.transpose(-2, -1) # (B, head_size, T)
        else:
            qkv = self.qkv(x)
            q, k, v = qkv.chunk(3, dim=-1)
            k_t = k.transpose(-2, -1) # (B, head_size, T)

        dots = torch.matmul(q, k_t) * (self.head_size ** -0.5) # (B, T, T)
        if self.use_mask:
            dots = dots.masked_fill(self.mask[:T, :T] == 0, float('-inf')) # (B, T, T)
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        # print(k_t.shape, v.shape)
        out = torch.matmul(attn, v) # (B, T, head_size)
        return out

class Cross_Attention(nn.Module):
    def __init__(self, dropout=0.1, use_kv_cache=False, device="cpu", dtype=torch.bfloat16, input_dim=1024, head_size=32, context_window=1024):
        super(Cross_Attention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.dtype = dtype
        self.head_size = head_size
        self.context_window = context_window
        self.kv_cache = None
        self.kv = nn.Linear(input_dim, 2 * head_size, dtype=dtype, bias=False)
        self.q = nn.Linear(input_dim, head_size, dtype=dtype, bias=False)
        self.to(device=device, dtype=dtype)
        self.use_kv_cache = use_kv_cache

    def forward(self, x_decoder, x_encoder):
        B, T, C = x_decoder.shape

        if self.use_kv_cache:
            if self.kv_cache is None:
                kf, vf = self.kv(x_encoder).chunk(2, dim=-1)
                kf, vf = kf.unsqueeze(1), vf.unsqueeze(1) # (B, 1, T, head_size) for the num heads dimension
                self.kv_cache = KVCache(batch_size=B,
                                        max_seq_len=self.context_window,
                                        num_kv_heads=1,
                                        head_dim=self.head_size,
                                        dtype=self.dtype
                                        )
                
                self.kv_cache = self.kv_cache.to(self.device)
                cached_k, cached_v =self.kv_cache.update(kf, vf)
                q = self.q(x_decoder) # (B, T, head_size)
            else:
                last_token = x_decoder[:, -1:, :] # (B, 1, C)
                q = self.q(last_token) # (B, 1, head_size)

                cached_k, cached_v = self.kv_cache.get() # (B, 1, T, head_size)
            cached_k, v = cached_k.squeeze(1), cached_v.squeeze(1) # (B, T, head_size)
            k_t = cached_k.transpose(-2, -1) # (B, head_size, T)
        else:
            q = self.q(x_decoder) # (B, T, head_size)
            kf, vf = self.kv(x_encoder).chunk(2, dim=-1)
            k_t = kf.transpose(-2, -1) # (B, head_size, T)
            v = vf
        dots = torch.matmul(q, k_t) * (self.head_size ** -0.5) # (B, T, T)
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)

        return out

class Multihead_Cross_Attention(nn.Module):
    def __init__(self, num_heads, n_embed, context_window, head_size, dropout=0.1, use_kv_cache=False, device="cpu", dtype=torch.bfloat16):
        super(Multihead_Cross_Attention, self).__init__()
        self.heads = nn.ModuleList()
        for _ in range(num_heads):
            self.heads.append(Cross_Attention(dropout=dropout, device=device, dtype=dtype, input_dim=n_embed, context_window=context_window, head_size=head_size, use_kv_cache=use_kv_cache))
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
        self.to(device=device, dtype=dtype)

    def forward(self, x_decoder, x_encoder):
        out = torch.cat([h(x_decoder, x_encoder) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class Multihead_Attention(nn.Module):
    def __init__(self, num_heads, n_embed, context_window, head_size, dropout=0.1, use_kv_cache=False, device="cpu", dtype=torch.bfloat16):
        super(Multihead_Attention, self).__init__()
        self.heads = nn.ModuleList()
        for _ in range(num_heads):
            self.heads.append(dot_product_attention(dropout=dropout, device=device, dtype=dtype, input_dim=n_embed, context_window=context_window, head_size=head_size, use_kv_cache=use_kv_cache))
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
        self.to(device=device, dtype=dtype)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, input_dim, dropout=0.1, device="cpu", dtype=torch.bfloat16):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim*4, dtype=dtype)
        self.fc2 = nn.Linear(input_dim*4, input_dim, dtype=dtype)
        self.dropout = nn.Dropout(dropout)
        self.to(device=device, dtype=dtype)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout((self.fc2(x)))
        return x

class TransformerBlock(nn.Module):
    def __init__(self, num_heads, n_embed, context_window, dropout=0.1, use_kv_cache=False, use_cross_att=False, device="cpu", dtype=torch.bfloat16):
        super(TransformerBlock, self).__init__()
        head_size = n_embed // num_heads
        if use_cross_att:
            self.cross_att = Multihead_Cross_Attention(num_heads, n_embed, context_window, head_size=head_size, dropout=dropout, device=device, dtype=dtype, use_kv_cache=use_kv_cache)
            self.ln3 = nn.LayerNorm(n_embed, dtype=dtype)
            self.ln4 = nn.LayerNorm(n_embed, dtype=dtype)
            self.ff2 = FeedForward(n_embed, dropout=dropout, device=device, dtype=dtype)
        self.attn = Multihead_Attention(num_heads, n_embed, context_window, head_size=head_size, dropout=dropout, device=device, dtype=dtype, use_kv_cache=use_kv_cache)
        self.ff = FeedForward(n_embed, dropout=dropout, device=device, dtype=dtype)
        self.ln1 = nn.LayerNorm(n_embed, dtype=dtype)
        self.ln2 = nn.LayerNorm(n_embed, dtype=dtype)
        self.to(device=device, dtype=dtype)
        self.use_cross_att = use_cross_att

    def forward(self, x, encoder_output=None):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        if self.use_cross_att:
            x = x + self.cross_att(self.ln3(x), encoder_output)
            x = x + self.ff2(self.ln4(x))
        return x

class PosEmbedding(nn.Module):
    def __init__(self, input_dim, context_window, device="cpu", dtype=torch.bfloat16):
        super(PosEmbedding, self).__init__()
        self.pos = torch.arange(0, context_window, dtype=dtype).unsqueeze(1)
        self.dim_idx = torch.arange(0, input_dim, 2, dtype=dtype)
        self.to(device=device, dtype=dtype)
        self.context_window = context_window
        self.input_dim = input_dim
        self.device = device
        self.dtype = dtype
        self.to(device=device, dtype=dtype)

        pe = torch.zeros(self.context_window, self.input_dim, dtype=self.dtype, device=self.device)
        div_term = torch.exp(self.dim_idx * (-math.log(10000.0) / self.input_dim))
        pe[:, 0::2] = torch.sin(self.pos * div_term)
        pe[:, 1::2] = torch.cos(self.pos * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        B, T, C = x.shape
        x = x + self.pe[:, :T]
        return x

class Network(nn.Module):
    def __init__(self, num_heads,
                        num_layer,
                        n_embed,
                        context_window, 
                        vocab_size,
                        dropout=0.1,
                        use_kv_cache=False,
                        use_encoder=False,
                        encoder=None,
                        encoder_context_window=None,
                        n_landmarks=None,
                        img_dim=None,
                        device="cpu",
                        dtype=torch.bfloat16):
        super(Network, self).__init__()
        self.emb = nn.Embedding(vocab_size, n_embed, dtype=dtype)
        self.pos_embed = PosEmbedding(n_embed, context_window, device=device, dtype=dtype)
        # self.pos_embed = nn.Embedding(context_window, n_embed, dtype=dtype)
        self.transformer = nn.ModuleList()
        for _ in range(num_layer):
            self.transformer.append(TransformerBlock(num_heads, n_embed, context_window, dropout=dropout, use_kv_cache=use_kv_cache, device=device, dtype=dtype))
        self.ln = nn.LayerNorm(n_embed, dtype=dtype)
        if use_encoder:
            if encoder is not None:
                self.encoder = encoder
            else:
                self.encoder = Transformer_Encoder(n_landmarks, img_dim, n_embed, num_heads, num_layer, encoder_context_window, device=device, dtype=dtype)
        self.out = nn.Linear(n_embed, vocab_size, dtype=dtype)
        # self.emb.weight = self.out.weight
        self.to(device=device, dtype=dtype)
        self.dtype = dtype
        self.device = device
        self.d_model = n_embed
        self.context_window = context_window
        self.use_encoder = use_encoder
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, landmarks=None, img=None):
        if landmarks is not None:
            landmarks = landmarks.to(device=self.device, dtype=self.dtype)
        if img is not None:
            img = img.to(device=self.device, dtype=self.dtype)
        x = x.to(device=self.device, dtype=self.dtype)
        B, T = x.shape
        # idx = torch.arange(T, device=self.device)
        if self.use_encoder:
            encoder_vec = self.encoder(img, landmarks)
        x = x.to(device=self.device, dtype=torch.int32)
        x = self.emb(x) * (self.d_model ** 0.5)
        x = self.pos_embed(x)

        for block in self.transformer:
            if self.use_encoder:
                x = block(x, encoder_output=encoder_vec)
            else:
                x = block(x)
        x = self.ln(x)
        # x = self.out(x[:, -1, :])
        x = self.out(x)
        return x

    def generate(self, x, max_length:int):
        if len(x.size()) == 1:
            x = x.unsqueeze(0)
        self.eval()
        with torch.no_grad():
            for _ in range(max_length):
                logits = self(x)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                token = torch.multinomial(probs, num_samples=1)
                x = torch.cat([x, token], dim=-1)
        return x
        
class SpatialEncoder(nn.Module):
    def __init__(self, image_dim:tuple[int ,int, int]=(32 ,256, 256)):
        super(SpatialEncoder, self).__init__()
        _, self.W, self.H = image_dim

        self.spat1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2, padding=1)
        self.spat2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.spat3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.spat4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.spat5 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.spat6 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        B, C, T, W, H = x.shape # 1, 3, 32, 256, 256
        original_T = T
        pad_amount = T % 2
        x = F.pad(x, (0, 0, 0, 0, 0, pad_amount))
        x = x.view(B*(T + pad_amount), C, W, H)
        x = self.spat1(x)
        x = self.relu(x)
        x = self.relu(self.spat2(x))
        x = self.spat3(x)
        x = self.relu(x)
        x = self.spat4(x)
        x = self.relu(x)
        x = x.view(B, 64, T + pad_amount, W//16, H//16)
        x = self.relu(self.spat5(x))
        x = self.relu(self.spat6(x)) # (B, C, T, W, H)
        B, C, T, W, H = x.shape
        x = x.view(B, T, C, W, H) # 1, 32, 128, 4, 4
        return x, original_T

class LandmarkCompression(nn.Module):
    def __init__(self, n_landmarks):
        super(LandmarkCompression, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=n_landmarks, out_channels=n_landmarks, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=n_landmarks, out_channels=n_landmarks, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        original_T = x.shape[1]
        x = x.transpose(-2, -1)
        pad1 = original_T % 2
        if pad1:
            x = F.pad(x, (0, pad1))
        x = self.relu(self.conv1(x))
        T1 = x.shape[-1]
        pad2 = T1 % 2
        if pad2:
            x = F.pad(x, (0, pad2))
        x = self.relu(self.conv2(x))
        x = x.transpose(-2, -1)
        return x, original_T

class Transformer_Encoder(nn.Module):
    def __init__(self, n_landmarks, img_dim, n_embed, num_heads, n_layer, encoder_context_window, device="cpu", dtype=torch.bfloat16, use_kv_cache=False):
        super(Transformer_Encoder, self).__init__()
        self.landmark_embed = nn.Linear(n_landmarks, n_embed//2)
        self.image_embed = nn.Linear(img_dim, n_embed//2)
        self.pos_embed = PosEmbedding(n_embed, encoder_context_window, device=device, dtype=dtype)
        self.mh_attention = nn.ModuleList([TransformerBlock(num_heads, n_embed, encoder_context_window, use_kv_cache=use_kv_cache, device=device, dtype=dtype) for _ in range(n_layer)])

        self.landmark_compression = LandmarkCompression(n_embed//2)
        self.img_compression = SpatialEncoder()

        self.ln = nn.LayerNorm(n_embed)

        self.out = nn.Linear(n_embed, n_embed*2)

        self.to(device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype
        
    def forward(self, img, landmarks):
        img = img.to(device=self.device, dtype=self.dtype)
        landmarks = landmarks.to(device=self.device, dtype=self.dtype)
        # B, C, T, W, H = img.shape
        # B, T, C = landmarks.shape
        compressed_img, original_T = self.img_compression(img) # (B, T, C, W, H)
        img_vec = compressed_img.flatten(start_dim=2) # (B, T, C * W * H)
        img_vec = self.image_embed(img_vec) # (B, T, n_embed/2)
        landmarks = self.landmark_embed(landmarks) # (B, T, n_embed/2)
        landmarks, landmark_T = self.landmark_compression(landmarks) # (B, T, n_embed/2)
        x = torch.cat([img_vec, landmarks], dim=-1) # (B, T, n_embed)
        x = self.pos_embed(x)
        x = self.ln(x)
        for block in self.mh_attention:
            x = block(x)
        x = self.out(x)
        mu, logvar = torch.chunk(x, 2, dim=-1)
        return mu, logvar, original_T, landmark_T

if __name__ == "__main__":
    from summary import ModelSummary
    with open("dataset.txt", "r") as f:
        text = f.read()
    tokenizer = CharacterLevelTokenizer().load("tokenizer.pkl")
    model = Network(num_heads=2,
                    num_layer=2,
                    n_embed=256,
                    context_window=4096,
                    vocab_size=len(tokenizer),
                    dropout=0.2,
                    use_kv_cache=True,
                    use_encoder=True,
                    encoder_context_window=512,
                    n_landmarks = 210,
                    img_dim=128 * 4 * 4,
                    device="cpu",
                    dtype=torch.bfloat16)
    # model.load_state_dict(torch.load("model.pt"))
    ModelSummary(model)
    x = torch.tensor(tokenizer.encode("C")).unsqueeze(0)
    img = torch.rand(1, 3, 1, 256, 256)
    landmarks = torch.rand(1, 1, 210)
    # print(x)
    y = model(x, landmarks, img)
    # print(y.shape)
    # seq = model.generate(x, max_length=500)
    # print(seq)
    # seq = list(seq[0].tolist())
    # print(seq)
    # print(tokenizer.decode(seq))