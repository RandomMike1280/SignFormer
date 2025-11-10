import torch
import torch.nn as nn
import torch.nn.functional as F
from network import TransformerBlock, PosEmbedding
from mixer import Mixer_block

class SpatialEncoder(nn.Module):
    def __init__(self, image_dim:tuple[int ,int, int]=(32 ,256, 256)):
        super(SpatialEncoder, self).__init__()
        _, self.W, self.H = image_dim

        self.spat1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2, padding=1)
        self.spat2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.spat_temp3 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.spat_temp4 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        B, C, T, W, H = x.shape # 1, 3, 32, 256, 256
        original_T = T
        pad_amount = T % 2
        x = F.pad(x, (0, 0, 0, 0, 0, pad_amount))
        x = x.view(B*(T + pad_amount), C, W, H)
        x = self.spat1(x) # (B*(T + pad_amount), 8, W//2, H//2)
        x = self.relu(x)
        x = self.relu(self.spat2(x)) # (B*(T + pad_amount), 16, W//4, H//4)
        x = x.view(B, 16, T + pad_amount, W//4, H//4)
        x = self.relu(self.spat_temp3(x)) # (B, 32, T//2, W//8, H//8)
        x = self.relu(self.spat_temp4(x)) # (B, 32, T//4, W//16, H//16)
        B, C, T, W, H = x.shape
        x = x.view(B, T, C, W, H) # B, T, 32, 16, 16
        return x, original_T

class SpatialDecoder(nn.Module):
    def __init__(self, image_dim:tuple[int ,int, int]=(32 ,256, 256)):
        super(SpatialDecoder, self).__init__()
        _, self.W, self.H = image_dim

        self.spat1 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.spat2 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.spat_temp3 = nn.ConvTranspose3d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.spat_temp4 = nn.ConvTranspose3d(in_channels=8, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.relu = nn.ReLU()

    def forward(self, x, original_T=None):
        B, T, C, W, H = x.shape # B, T, 32, 16, 16
        x = x.view(B*T, C, W, H)
        x = self.spat1(x) # (B*T, 32, W*2, H*2)
        x = self.relu(x)
        x = self.relu(self.spat2(x)) # (B*T, 16, W*4, H*4)
        x = self.relu(self.spat_temp3(x)) # (B*T, 8, W*8, H*8)
        x = self.relu(self.spat_temp4(x)) # (B*T, 3, W*16, H*16)
        x = x.view(B, T, C, W*16, H*16)
        B, C, T, W, H = x.shape
        x = x.view(B, T, C, W, H) # B, T, 3, 256, 256
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
        self.pos_embed = PosEmbedding(n_embed, encoder_context_window, device=device, dtype=dtype)
        self.mh_attention = nn.ModuleList([TransformerBlock(num_heads, n_embed, encoder_context_window, use_kv_cache=use_kv_cache, use_mask=False, device=device, dtype=dtype) for _ in range(n_layer)])

        self.landmark_compression = LandmarkCompression(n_landmarks)
        self.img_compression = SpatialEncoder()
        self.mixer = Mixer_block(img_channels=32, img_dim=16, patch_size=4, hidden_dim=n_embed)

        self.ln = nn.LayerNorm(n_embed)

        self.out = nn.Linear(n_embed, n_embed)

        self.normalize = nn.LayerNorm(n_landmarks)

        self.to(device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype
        
    def forward(self, img, landmarks):
        ### WIP
        img = img.to(device=self.device, dtype=self.dtype)
        landmarks = landmarks.to(device=self.device, dtype=self.dtype)
        # B, C, T, W, H = img.shape
        # B, T, C = landmarks.shape
        landmarks = self.normalize(landmarks)
        compressed_img, original_T = self.img_compression(img) # (B, T//4, C, W//16, H//16)
        img_tokens = self.mixer(compressed_img) # (B, T*P, C)
        landmarks, landmark_T = self.landmark_compression(landmarks) # (B, T//4, n_landmarks)
        landmarks = self.landmark_embed(landmarks) # (B, T//4, n_embed/2)
        x = torch.cat([img_vec, landmarks], dim=-1) # (B, T//4, n_embed)
        x = self.pos_embed(x)
        x = self.ln(x)
        for block in self.mh_attention:
            x = block(x)
        x = self.out(x)
        return x, original_T, landmark_T

class Transformer_Decoder(nn.Module):
    def __init__(self, n_landmarks, img_dim, n_embed, num_heads, n_layer, encoder_context_window, device="cpu", dtype=torch.bfloat16, use_kv_cache=False):
        super(Transformer_Decoder, self).__init__()
        self.mh_attention = nn.ModuleList([TransformerBlock(num_heads, n_embed, encoder_context_window, use_kv_cache=use_kv_cache, use_mask=False, device=device, dtype=dtype) for _ in range(n_layer)])
        self.landmark_out = nn.Linear(n_embed, n_landmarks)
        self.image_out = nn.Linear(n_embed, img_dim)
        self.img_decompression = SpatialDecoder()
        self.landmark_decompression = LandmarkDecompression(n_landmarks)
        self.pos_embed = PosEmbedding(n_embed, encoder_context_window, device=device, dtype=dtype)

        self.device = device
        self.dtype = dtype
        
        self.to(device=device, dtype=dtype)
        
    def forward(self, x, original_T, landmark_T):
        ### WIP
        x = self.pos_embed(x)
        for block in self.mh_attention:
            x = block(x)
        landmarks = self.landmark_out(x)
        landmarks = self.landmark_decompression(landmarks, landmark_T)
        img = self.image_out(x)
        img = img.view(x.shape[0], x.shape[1], 128, 16, 16)
        img = self.img_decompression(img, original_T)
        return landmarks, img

class VAE(nn.Module):
    def __init__(self, encoder, decoder, device="cpu", dtype=torch.bfloat16):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.to(device=torch.device(device), dtype=dtype)

    def forward(self, video):
        z, img_T, lm_T = self.encoder(video)

        recon = self.decoder(z, img_T, lm_T)

        return recon

    def get_models(self):
        return self.encoder, self.decoder

if __name__ == "__main__":
    from summary import ModelSummary
    model = Transformer_Encoder(n_landmarks=1106, img_dim=32*16*16, n_embed=256, num_heads=2, n_layer=2, encoder_context_window=4096, device="cpu", dtype=torch.bfloat16)
    ModelSummary(model)
    img = torch.rand(1, 3, 33, 256, 256)
    landmarks = torch.rand(1, 33, 1106)
    mu, logvar, img_T, lm_T = model(img, landmarks)
    print(mu.shape, logvar.shape)
    decoder = Transformer_Decoder(n_landmarks=1106, img_dim=32*16*16, n_embed=256, num_heads=2, n_layer=2, encoder_context_window=4096, device="cpu", dtype=torch.bfloat16)
    ModelSummary(decoder)
    out = decoder(mu, img_T, lm_T)
    print(out[0].shape, out[1].shape)