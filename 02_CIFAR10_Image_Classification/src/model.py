"""
CIFAR-10 Image Classification - Deep CNN Model Architecture
Author: Rudhra Sitholey (Reg No: 23BCY10296 | App No: IN26012560)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Convolutional Block with Batch Normalization and ReLU Activation.
    """
    def __init__(self, in_channels, out_channels, pool=False):
        super(ConvBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResNetCIFAR(nn.Module):
    """
    Deep Residual Convolutional Neural Network for CIFAR-10 classification.
    """
    def __init__(self, num_classes=10):
        super(ResNetCIFAR, self).__init__()
        
        self.prep = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.layer1 = ConvBlock(64, 128, pool=True)
        self.layer2 = ConvBlock(128, 256, pool=True)
        self.layer3 = ConvBlock(256, 512, pool=True)
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.classifier(x)
        return x
