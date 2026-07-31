"""
LFW Face Recognition - Deep Feature Extractor
Author: Rudhra Sitholey (Reg No: 23BCY10296 | App No: IN26012560)
"""

import torch
import torch.nn as nn


class FaceEmbeddingNet(nn.Module):
    """
    CNN Architecture generating 128-dimensional L2-normalized embeddings for faces.
    """
    def __init__(self, embedding_dim=128):
        super(FaceEmbeddingNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, x):
        features = self.features(x)
        embeddings = self.fc(features)
        # L2 Normalization for Cosine distance matching
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings
