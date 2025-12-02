import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedFusion(nn.Module):
    def __init__(self, channels=256, num_modalities=3):
        super().__init__()
        self.num_modalities = num_modalities
        self.channels = channels
        self.gate_fc = nn.Linear(channels * num_modalities, num_modalities)
    
    def forward(self, feats):
        assert len(feats) == self.num_modalities, \
            f"Expected {self.num_modalities} feature maps but got {len(feats)}"
        B, C, H, W = feats[0].shape
        pooled = [f.mean(dim=[2, 3]) for f in feats]
        concat = torch.cat(pooled, dim=1)
        weights = self.gate_fc(concat)
        weights = F.softmax(weights, dim=1)

        fused = 0
        for i in range(self.num_modalities):
            w = weights[:, i].view(B, 1, 1, 1)
            fused += w * feats[i]
        return fused

if __name__ == '__main__':
    fusion = GatedFusion(channels=256, num_modalities=3)

    feat1 = torch.randn(2, 256, 75, 75)
    feat2 = torch.randn(2, 256, 75, 75)
    feat3 = torch.randn(2, 256, 75, 75)

    F = fusion([feat1, feat2, feat3])
    print(F.shape)