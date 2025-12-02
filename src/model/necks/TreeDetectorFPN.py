import torch
import torch.nn as nn
import torch.nn.functional as F

class TreeDetectorFPN(nn.Module):
    def __init__(self, in_channels=256, out_channels=256):
        super().__init__()
        
        self.p3_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.p4_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.p5_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        
        self.gn3 = nn.GroupNorm(32, out_channels)
        self.gn4 = nn.GroupNorm(32, out_channels)
        self.gn5 = nn.GroupNorm(32, out_channels)

    def forward(self, x):
        p3 = F.relu(self.gn3(self.p3_conv(x)))
        p4 = F.relu(self.gn4(self.p4_conv(p3)))
        p5 = F.relu(self.gn5(self.p5_conv(p4)))
        return [p3, p4, p5]