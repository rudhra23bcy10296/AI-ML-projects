"""
Cancer Detection using MRI - Transfer Learning Deep Architecture
Author: Rudhra Sitholey (Reg No: 23BCY10296 | App No: IN26012560)
"""

import torch
import torch.nn as nn
import torchvision.models as models


class MRICancerDetector(nn.Module):
    """
    Transfer learning network for MRI tumor classification.
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(MRICancerDetector, self).__init__()
        
        # ResNet18 backbone
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        in_features = self.backbone.fc.in_features
        
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)
