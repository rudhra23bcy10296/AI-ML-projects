"""
Cancer Detection using MRI - Image Preprocessing & CLAHE Module
Author: Rudhra Sitholey (Reg No: 23BCY10296 | App No: IN26012560)
"""

import numpy as np
import torch
import cv2


def apply_clahe(image_np):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to MRI scans.
    """
    if len(image_np.shape) == 3 and image_np.shape[0] == 3:
        image_np = np.transpose(image_np, (1, 2, 0))
        
    gray = cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB) / 255.0
    return torch.tensor(enhanced_rgb, dtype=torch.float32).permute(2, 0, 1)


def generate_synthetic_mri_dataset(num_samples=240, img_size=(3, 128, 128)):
    """
    Generates synthetic MRI brain scan dataset (Tumor Positive vs Normal Negative).
    """
    np.random.seed(42)
    torch.manual_seed(42)
    
    data = torch.randn(num_samples, *img_size)
    # Add synthetic lesion signal to positive cases
    labels = torch.randint(0, 2, (num_samples,))
    for i in range(num_samples):
        if labels[i] == 1:
            data[i, :, 45:75, 45:75] += 2.5  # Simulate high-intensity tumor signal
            
    return data, labels
