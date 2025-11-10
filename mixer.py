import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1, device='cpu', dtype=torch.bfloat16):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=False)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.to(device=device, dtype=dtype)
    
    def forward(self, x):
        x = self.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class MLP_Mixer(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_patches, device='cpu', dtype=torch.bfloat16):
        super(MLP_Mixer, self).__init__()
        self.hidden_dim = hidden_dim

        self.mlp1 = MLP(n_patches, n_patches, n_patches)
        self.mlp2 = MLP(input_dim, hidden_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)

        self.to(device=device, dtype=dtype)

    def forward(self, x):
        x_skipped = x
        x = self.ln(x)
        x = x.transpose(-2, -1)
        x = self.mlp1(x)
        x = x.transpose(-2, -1) + x_skipped # (B, P, C)
        x = self.mlp2(self.ln(x))
        return x

class Mixer_block(nn.Module):
    def __init__(self, img_channels, img_dim, patch_size, hidden_dim, device='cpu', dtype=torch.bfloat16):
        super(Mixer_block, self).__init__()
        self.img_channels = img_channels
        self.img_dim = img_dim
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.n_patches = img_dim**2 // patch_size**2

        self.pp_fc = nn.Linear(patch_size**2 * img_channels, hidden_dim)
        self.mlp_mixer = nn.ModuleList([
            MLP_Mixer(hidden_dim, hidden_dim, self.n_patches, device=device, dtype=dtype)
            for _ in range(2)
        ])

        self.device = device
        self.dtype = dtype
        self.to(device=device, dtype=dtype)

    def patchify(self, x: torch.Tensor):
        """
        Given a C x H x W image, return N patches of size n_patches x n_patches
        where N = H * W / (n_patches ** 2)
        The returned patches have the shape of (B, T, C, W, H)
        Note: The input 'x' is assumed to be a single image (C, H, W).
              To align with the requested output shape (B, T, C, W, H),
              we'll assume B=1 for a single input image and T will be N.
              The final W and H of the patches will be self.patch_size.
        """
        B, T, C, H, W = x.shape
        patch_size = self.patch_size

        if H % patch_size != 0 or W % patch_size != 0:
            raise ValueError(
                f"Image dimensions ({H}x{W}) must be divisible by "
                f"patch size ({patch_size})"
            )

        # Reshape the image to extract patches
        # C, H, W -> C, num_patches_h, patch_size, num_patches_w, patch_size
        patches = (
            x.unfold(1, patch_size, patch_size)  # Unfold height
            .unfold(2, patch_size, patch_size)  # Unfold width
            .reshape(
                B, T, C, self.n_patches, patch_size, patch_size
            )  # C, N, P_H, P_W
        )

        # Transpose to get the desired (B, T, C, W, H) shape
        # where B=1, P=N, C=C, W=patch_size, H=patch_size
        patches = patches.permute(
            0, 1, 3, 2, 4, 5
        )  # B, T, P, C, W, H

        return patches

    def forward(self, x):
        patches = self.patchify(x) # (B, T, P, C, W, H)
        B, T, P, C, W, H = patches.shape
        x = x.reshape(B * T, P, C * W * H)
        x = self.pp_fc(x)
        for mixer in self.mlp_mixer:
            x = mixer(x)
        x = x.view(B, T*P, self.hidden_dim)
        return x

if __name__ == "__main__":
    img = torch.randn(1, 32, 32, 16, 16)
    mixer = Mixer_block(img_channels=32, img_dim=16, patch_size=4, hidden_dim=512, device='cpu', dtype=torch.float32)
    patches = mixer.patchify(img)
    print(patches.shape)
    y = mixer(img)
    print(y.shape)