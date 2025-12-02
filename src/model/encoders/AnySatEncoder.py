import pyrootutils
root_path = pyrootutils.setup_root(__file__, indicator='.git', pythonpath=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils.utils import get_divice, timestamp2doy

class AnySatEncoder(nn.Module):
    def __init__(self, 
                 aerial, 
                 s1, 
                 s2, 
                 device, 
                 patch_size=10, 
                 output="dense", 
                 output_modality="aerial",
                 flash_attention=False):
        super().__init__()
        self.aerial = aerial
        self.s1 = s1
        self.s2 = s2
        assert s1.shape[0] == s2.shape[0] and s1.shape[1] == s2.shape[1], "The batch size of timestamps of Sentinel data should be same"
        self.batch_size = s1.shape[0]
        self.timestamps = s1.shape[1]
        self.device = device

        self.sentinel_dates = timestamp2doy(self.timestamps, self.batch_size, self.device)
        self.anysat_model = torch.hub.load('gastruc/anysat', 'anysat', pretrained=True, flash_attn=flash_attention)
        self.anysat_model.to(self.device)

        for p in self.anysat_model.parameters():
            p.requires_grad = False

        self.anysat_model.eval()

        self.patch_size = patch_size
        self.output = output
        self.output_modality = output_modality
        
    def forward(self):
        data = {
            "aerial": self.aerial.to(self.device),
            "s2": self.s2.to(self.device),
            "s2_dates": self.sentinel_dates,
            "s1": self.s1.to(self.device),
            "s1_dates": self.sentinel_dates,
        }
    
        with torch.no_grad():
            features = self.anysat_model(data, patch_size=self.patch_size, output=self.output, output_modality=self.output_modality)
        return features 

    
if __name__ == '__main__':
    device = get_divice()
    B = 4   # batch size

    s2_images = torch.randn(B, 4, 10, 6, 6).float()
    s1_images = torch.randn(B, 4, 3, 6, 6).float()
    aerial_images = torch.randn(B, 4, 300, 300).float()

    model = AnySatEncoder(aerial=aerial_images, s1=s1_images, s2=s2_images, device=device)
    model = model.to(device=device)
    result = model.forward()
    print(result.shape)