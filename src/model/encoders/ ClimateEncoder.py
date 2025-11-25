import pyrootutils
root_path = pyrootutils.setup_root(__file__, indicator='.git', pythonpath=True)

import torch
import torch.nn as nn
from rshf.climplicit import Climplicit
from Utils.utils import get_divice


class ClimateEncoder(nn.Module):
    def __init__(self, 
                 H: int=80, 
                 W: int=80,
                 without_month=True,
                 defalut_month: int=8, 
                 embed_dim: int=256,
                 return_chelsa=False,
                 frozen=True,
                ):
        
        super().__init__()
        self.H = H
        self.W = W
        self.default_month = defalut_month
        self.embed_dim = embed_dim
        self.return_chelsa = return_chelsa
        self.without_month = without_month
        self.frozen = frozen
        self.clim_model = Climplicit.from_pretrained("Jobedo/climplicit", 
                                                     config={"return_chelsa": self.return_chelsa})
        
        in_dim = 1024 if self.without_month else 256
        
        if in_dim == embed_dim:
            self.out_proj = nn.Identity()
        else:
            self.out_proj = nn.Linear(in_dim, embed_dim)
        
        if self.frozen:
            for p in self.clim_model.parameters():
                p.requires_grad = False
            self.clim_model.eval()

    def forward(self, loc, month=None):
        B = loc.size(0) # Batch size

        if not self.without_month:
            if month is None:
                month = torch.full((B,), self.default_month, dtype=torch.float32, device=loc.device)
            else:
                month = torch.full((B,), month, dtype=torch.float32, device=loc.device)
            clim_emb = self.clim_model(loc, month)
        else:
            clim_emb = self.clim_model(loc)
        
        clim_emb = self.out_proj(clim_emb)
        feature_map = clim_emb[:, :, None, None].expand(-1, -1, self.H, self.W)
        
        return feature_map


if __name__ == '__main__':
    model = ClimateEncoder(H=75, W=75, without_month=True)
    device = get_divice()
    model = model.to(device)
    loc = [8.550155, 47.396702]
    result = model.forward(torch.tensor([loc] * 125).to(device))
    print(result.shape)
