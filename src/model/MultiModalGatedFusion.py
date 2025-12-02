import pyrootutils
root_path = pyrootutils.setup_root(__file__, indicator='.git', pythonpath=True)

import torch
import torch.nn as nn

from src.model.encoders.AnySatEncoder import AnySatEncoder
from src.model.encoders.VegHeightModelEncoder import VegHeightModelEncoder
from src.model.encoders.ClimateEncoder import ClimateEncoder
from src.model.GatedFusion import GatedFusion


class MultiModalGatedFusionModel(nn.Module):
    def __init__(self, 
                 device,
                 fusion_channels=256,
                 removed_modality=None):
        super().__init__()

        self.device = device
        self.removed_modality = removed_modality

        self.encoder_anysat = AnySatEncoder(device=device).to(device)
        self.encoder_vhm = VegHeightModelEncoder().to(device)
        self.encoder_climate = ClimateEncoder().to(device)

        if self.removed_modality is None:
            num_modalities = 3
        else:
            num_modalities = 2

        self.fusion = GatedFusion(channels=fusion_channels,
                                  num_modalities=num_modalities)


    def forward(self, data):
        aerial = data['image'].to(self.device)
        vhm = data['vhm'].to(self.device)
        s1 = data['s1_ts'].to(self.device)
        s2 = data['s2_ts'].to(self.device)
        loc = data['loc'].to(self.device)

        feat_anysat = self.encoder_anysat(aerial=aerial, s1=s1, s2=s2)

        modality_feats = [feat_anysat]

        if self.removed_modality != "vhm":
            feat_vhm = self.encoder_vhm(vhm)
            modality_feats.append(feat_vhm)

        if self.removed_modality != "climate":
            feat_climate = self.encoder_climate(loc)
            modality_feats.append(feat_climate)

        fused_feat = self.fusion(modality_feats)

        return fused_feat
            
            





