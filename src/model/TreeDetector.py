import pyrootutils
root_path = pyrootutils.setup_root(__file__, indicator='.git', pythonpath=True)

import torch
import torch.nn as nn

from src.model.MultiModalGatedFusion import MultiModalGatedFusionModel
from src.model.necks.TreeDetectorFPN import TreeDetectorFPN
from src.model.detection_head import FCOSHead 

class TreeDetector(nn.Module):
    def __init__(self, 
                 device, 
                 num_classes, 
                 fusion_channels=256, 
                 removed_modality=None):
        super().__init__()
        
        self.backbone = MultiModalGatedFusionModel(
            device=device,
            fusion_channels=fusion_channels,
            removed_modality=removed_modality
        )

        self.neck = TreeDetectorFPN(
            in_channels=fusion_channels,
            out_channels=fusion_channels
        )

        self.head = FCOSHead(
            num_classes=num_classes,
            in_channels=fusion_channels,
            num_levels=3 
        )

    def forward(self, data):
        feature_map = self.backbone(data)

        fpn_features = self.neck(feature_map)

        output = self.head(fpn_features)
        
        return output


if __name__ == '__main__':
    from Utils.utils import get_divice
    device = get_divice()

    dummy_data = {
        'image': torch.randn(2, 4, 300, 300).to(device),
        'vhm': torch.randn(2, 1, 120, 120).to(device),
        's1_ts': torch.randn(2, 12, 3, 6, 6).to(device),
        's2_ts': torch.randn(2, 12, 10, 6, 6).to(device),
        'loc': torch.randn(2, 2).to(device)
    }
    
    model = TreeDetector(device=device, num_classes=12).to(device)
    output = model(dummy_data)