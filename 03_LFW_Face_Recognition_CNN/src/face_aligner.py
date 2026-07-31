"""
LFW Face Recognition - Face Alignment & Preprocessing Utility
Author: Rudhra Sitholey (Reg No: 23BCY10296 | App No: IN26012560)
"""

import numpy as np
import torch


def preprocess_face_images(num_samples=200, img_size=(3, 64, 64), num_identities=10):
    """
    Simulates aligned face image tensors and target identity labels from LFW wild conditions.
    """
    np.random.seed(42)
    torch.manual_seed(42)
    
    images = torch.randn(num_samples, *img_size)
    labels = torch.randint(0, num_identities, (num_samples,))
    
    identity_names = [f"Person_{i+1}" for i in range(num_identities)]
    return images, labels, identity_names
