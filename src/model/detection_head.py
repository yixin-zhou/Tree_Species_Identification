import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self, x):
        return x * self.scale


class FCOSHead(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 256,
        feat_channels: int = 256,
        num_convs: int = 4,
        num_levels: int = 3,
        use_gn: bool = True,
        prior_prob: float = 0.01
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_levels = num_levels
        
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        
        for i in range(num_convs):
            self.cls_convs.append(
                nn.Conv2d(in_channels if i == 0 else feat_channels,
                          feat_channels, kernel_size=3, stride=1, padding=1, bias=False)
            )
            self.reg_convs.append(
                nn.Conv2d(in_channels if i == 0 else feat_channels,
                          feat_channels, kernel_size=3, stride=1, padding=1, bias=False)
            )
            
            if use_gn:
                self.cls_convs.append(nn.GroupNorm(32, feat_channels))
                self.reg_convs.append(nn.GroupNorm(32, feat_channels))
            
            self.cls_convs.append(nn.ReLU(inplace=True))
            self.reg_convs.append(nn.ReLU(inplace=True))

        self.cls_tower = nn.Sequential(*self.cls_convs)
        self.reg_tower = nn.Sequential(*self.reg_convs)

        self.cls_logits = nn.Conv2d(feat_channels, num_classes, kernel_size=3, stride=1, padding=1)

        self.bbox_pred = nn.Conv2d(feat_channels, 4, kernel_size=3, stride=1, padding=1)

        self.centerness = nn.Conv2d(feat_channels, 1, kernel_size=3, stride=1, padding=1)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(num_levels)])

        self._init_weights(prior_prob)

    def _init_weights(self, prior_prob):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, feats):
        assert len(feats) == self.num_levels, f"Expect {self.num_levels} levels, got {len(feats)}"
        
        all_cls_logits = []
        all_bbox_reg = []
        all_centerness = []

        for lvl, x in enumerate(feats):
            cls_feat = self.cls_tower(x)
            reg_feat = self.reg_tower(x)

            cls_logits = self.cls_logits(cls_feat)

            centerness = self.centerness(reg_feat)

            bbox_reg = self.bbox_pred(reg_feat)

            bbox_reg = self.scales[lvl](bbox_reg)
            
            if self.training:
                bbox_reg = torch.clamp(bbox_reg, max=10.0) 
            
            bbox_reg = torch.exp(bbox_reg)

            all_cls_logits.append(cls_logits)
            all_bbox_reg.append(bbox_reg)
            all_centerness.append(centerness)

        return {
            "cls_logits": all_cls_logits,
            "bbox_reg": all_bbox_reg,
            "centerness": all_centerness
        }