"""
Cancer Detection using MRI - Grad-CAM Visual Explainability Module
Author: Rudhra Sitholey (Reg No: 23BCY10296 | App No: IN26012560)
"""

import torch
import numpy as np


class GradCAM:
    """
    Computes Gradient-weighted Class Activation Mapping (Grad-CAM) heatmaps
    to highlight diagnostic regions in MRI scans.
    """
    def __init__(self, model, target_layer_name='layer4'):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        
        # Hook registration
        for name, module in self.model.named_modules():
            if target_layer_name in name and isinstance(module, torch.nn.Conv2d):
                module.register_forward_hook(self.save_activation)
                module.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_image, class_idx=None):
        output = self.model(input_image)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
            
        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()
        
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)
        
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx
