import torch
import torch.nn as nn

class SpatialEncoder(nn.Module):
    def __init__(self, image_dim:tuple[int ,int, int]=(32 ,256, 256)):
        super(SpatialEncoder, self).__init__()
        _, self.W, self.H = image_dim

        self.spat1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2, padding=1)
        self.spat2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.spat3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.spat4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.spat5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.spat6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        B, C, T, W, H = x.shape
        x = x.view(B*T, C, W, H)
        x = self.spat1(x)
        x = self.relu(x)
        x = self.relu(self.spat2(x))
        x = self.spat3(x)
        x = self.relu(x)
        x = self.spat4(x)
        x = self.relu(x)
        x = self.relu(self.spat5(x))
        x = self.relu(self.spat6(x))
        x = x.view(B, T, 256, W//64, H//64)
        return x

model = SpatialEncoder()

dummy_input = torch.rand(4, 3, 32, 256, 256)
output = model(dummy_input)
print(output)
print(output.shape)
print(output.flatten(start_dim=2).shape)