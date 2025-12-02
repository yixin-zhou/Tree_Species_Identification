import pyrootutils
root_path = pyrootutils.setup_root(__file__, indicator='.git', pythonpath=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils.utils import get_divice

class VegHeightModelEncoder(nn.Module):
    def __init__(self,
                 in_height: int=120,
                 in_width: int=120,
                 out_height: int=75,
                 out_width: int=75,
                 embed_dim=256,
                 ):
        super().__init__()
        
        self.in_height = in_height
        self.in_width = in_width
        self.embed_dim = embed_dim

        self.out_h = out_height
        self.out_w = out_width

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.LayerNorm([64, 60, 60]),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.LayerNorm([128, 60, 60]),

            nn.Conv2d(128, self.embed_dim, kernel_size=1, stride=1),
            nn.GELU(),
            nn.LayerNorm([self.embed_dim, 60, 60]),
        )
    
    def forward(self, x):
        assert x.dim() == 4, "The dim of VHM_Encoder input should be 4"
        feature_60 = self.encoder(x)
        feature_75 = F.interpolate(feature_60, 
                                   size=(self.out_h, self.out_w),
                                   mode="bilinear", 
                                   align_corners=False
        )
        
        return feature_75

    
if __name__ == '__main__':
    device = get_divice()
    model = VegHeightModelEncoder().to(device)

    B = 8
    test_vhm = torch.randn(B, 1, 120, 120).to(device)
    out = model(test_vhm)

    print("Input shape :", test_vhm.shape)
    print("Output shape:", out.shape)

    out.mean().backward()
    print("Backward pass OK!")

        
        