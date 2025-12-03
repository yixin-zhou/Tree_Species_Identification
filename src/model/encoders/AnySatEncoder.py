import pyrootutils
root_path = pyrootutils.setup_root(__file__, indicator='.git', pythonpath=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils.utils import get_divice, timestamp2doy

class AnySatEncoder(nn.Module):
    def __init__(self, device, patch_size=10, output="dense", output_modality="aerial", flash_attention=False):
        super().__init__()
        self.device = device

        self.anysat_model = torch.hub.load(
            'gastruc/anysat',
            'anysat',
            pretrained=True,
            flash_attn=flash_attention
        ).to(device)

        for p in self.anysat_model.parameters():
            p.requires_grad = False

        self.anysat_model.eval()

        self.patch_size = patch_size
        self.output = output
        self.output_modality = output_modality

        self.reduce_conv = nn.Conv2d(1536, 256, kernel_size=1)
        self.out_size = (75, 75)
    
    def forward(self, aerial, s1, s2):
        timestamps = s1.shape[1]
        sent_dates = timestamp2doy(timestamps, aerial.shape[0], aerial.device)

        data = {
            "aerial": aerial,
            "s1-asc": s1,
            "s1-asc_dates": sent_dates,
            "s2": s2,
            "s2_dates": sent_dates,
        }

        with torch.no_grad():
            features = self.anysat_model(
                data,
                patch_size=self.patch_size,
                output=self.output,
                output_modality=self.output_modality
            )

        features = features.permute(0, 3, 1, 2)
        features = self.reduce_conv(features)
        features = F.interpolate(features, size=self.out_size, mode="bilinear", align_corners=False)

        return features

    
if __name__ == '__main__':
    device = get_divice()
    B = 4   # batch size

    s2_images = torch.randn(B, 4, 10, 6, 6).float().to(device)
    s1_images = torch.randn(B, 4, 2, 6, 6).float().to(device)
    aerial_images = torch.randn(B, 4, 300, 300).float().to(device)

    model = AnySatEncoder(device=device)
    model = model.to(device=device)
    result = model.forward(aerial=aerial_images, s1=s1_images, s2=s2_images)
    print(result.shape)