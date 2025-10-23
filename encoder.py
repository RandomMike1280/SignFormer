import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from network import TransformerBlock

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

class SpatialDecoder(nn.Module):
    def __init__(self, image_dim:tuple[int ,int, int]=(32 ,256, 256)):
        super(SpatialDecoder, self).__init__()
        _, self.W, self.H = image_dim

        self.spat1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.spat2 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.spat3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.spat4 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.spat5 = nn.ConvTranspose3d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.spat6 = nn.ConvTranspose3d(in_channels=8, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.relu = nn.ReLU()

    def forward(self, x, original_T=None):
        B, T, C, W, H = x.shape # 1, 32, 128, 4, 4
        x = x.view(B*T, C, W, H)
        x = self.spat1(x)
        x = self.relu(x)
        x = self.relu(self.spat2(x))
        x = self.spat3(x)
        x = self.relu(x)
        x = self.spat4(x)
        x = self.relu(x)
        x = x.view(B, 16, T, W*16, H*16)
        x = self.relu(self.spat5(x))
        x = self.relu(self.spat6(x)) # (B, C, T, W, H)
        B, C, T, W, H = x.shape
        x = x.view(B, T, C, W, H) # 1, 32, 128, 256, 256
        if original_T is not None:
            x = x[:, :original_T, :, :, :]
        return x

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

class LandmarkDecompression(nn.Module):
    def __init__(self, n_landmarks):
        super(LandmarkDecompression, self).__init__()
        self.deconv1 = nn.ConvTranspose1d(in_channels=n_landmarks, out_channels=n_landmarks, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose1d(in_channels=n_landmarks, out_channels=n_landmarks, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu = nn.ReLU()

    def forward(self, x, original_T=None):
        x = x.transpose(-2, -1)
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = x.transpose(-2, -1)
        if original_T is not None:
            x = x[:, :original_T, :]
        return x

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

        self.normalize = nn.LayerNorm(n_landmarks)

        self.to(device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype
        
    def forward(self, img, landmarks):
        img = img.to(device=self.device, dtype=self.dtype)
        landmarks = landmarks.to(device=self.device, dtype=self.dtype)
        # B, C, T, W, H = img.shape
        # B, T, C = landmarks.shape
        landmarks = self.normalize(landmarks)
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

class Transformer_Decoder(nn.Module):
    def __init__(self, n_landmarks, img_dim, n_embed, num_heads, n_layer, encoder_context_window, device="cpu", dtype=torch.bfloat16, use_kv_cache=False):
        super(Transformer_Decoder, self).__init__()
        self.mh_attention = nn.ModuleList([TransformerBlock(num_heads, n_embed, encoder_context_window, use_kv_cache=use_kv_cache, device=device, dtype=dtype) for _ in range(n_layer)])
        self.landmark_out = nn.Linear(n_embed, n_landmarks)
        self.image_out = nn.Linear(n_embed, img_dim)
        self.img_decompression = SpatialDecoder()
        self.landmark_decompression = LandmarkDecompression(n_landmarks)
        self.pos_embed = PosEmbedding(n_embed, encoder_context_window, device=device, dtype=dtype)

        self.device = device
        self.dtype = dtype
        
        self.to(device=device, dtype=dtype)
        
    def forward(self, x, original_T, landmark_T):
        x = self.pos_embed(x)
        for block in self.mh_attention:
            x = block(x)
        landmarks = self.landmark_out(x)
        landmarks = self.landmark_decompression(landmarks, landmark_T)
        img = self.image_out(x)
        img = img.view(x.shape[0], x.shape[1], 128, 4, 4)
        img = self.img_decompression(img, original_T)
        return landmarks, img

class VAE(nn.Module):
    def __init__(self, encoder, decoder, device="cpu", dtype=torch.bfloat16):
        self.encoder = encoder
        self.decoder = decoder

        self.to(device=torch.device(device), dtype=dtype)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, video):
        mu, logvar = self.encoder(video)

        z = self.reparameterize(mu, logvar)

        recon = self.decoder(Z)

        return recon, mu, logvar

    def get_models(self):
        return self.encoder, self.decoder

if __name__ == "__main__":
    from summary import ModelSummary
    model = Transformer_Encoder(n_landmarks=1106, img_dim=128*4*4, n_embed=256, num_heads=2, n_layer=2, encoder_context_window=4096, device="cpu", dtype=torch.bfloat16)
    ModelSummary(model)
    img = torch.rand(1, 3, 33, 256, 256)
    landmarks = torch.rand(1, 33, 1106)
    mu, logvar, img_T, lm_T = model(img, landmarks)
    print(mu.shape, logvar.shape)
    decoder = Transformer_Decoder(n_landmarks=1106, img_dim=128*4*4, n_embed=256, num_heads=2, n_layer=2, encoder_context_window=4096, device="cpu", dtype=torch.bfloat16)
    ModelSummary(decoder)
    out = decoder(mu, img_T, lm_T)
    print(out[0].shape, out[1].shape)