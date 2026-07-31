"""
Cancer Detection using MRI Images - Main Execution Script
Author: Rudhra Sitholey (Reg No: 23BCY10296 | App No: IN26012560)
"""

import torch
from torch.utils.data import TensorDataset, DataLoader
from src.image_preprocessing import generate_synthetic_mri_dataset
from src.model import MRICancerDetector
from src.gradcam import GradCAM
from src.train_eval import train_mri_model, evaluate_mri_diagnostics


def main():
    print("=" * 65)
    print(" Project 4: Cancer Detection using MRI images")
    print(" Student: Rudhra Sitholey | Reg: 23BCY10296 | App: IN26012560")
    print("=" * 65)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[1] Computation Device: {device}")
    
    # 1. Dataset Generation
    print("\n[2] Loading Preprocessed MRI Scan Dataset...")
    data, labels = generate_synthetic_mri_dataset(num_samples=200)
    
    train_dataset = TensorDataset(data[:150], labels[:150])
    test_dataset = TensorDataset(data[150:], labels[150:])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    print(f"    Train Samples: {len(train_dataset)} | Test Samples: {len(test_dataset)}")
    
    # 2. Transfer Learning Model Setup
    print("\n[3] Initializing ResNet-18 Medical Vision Backbone...")
    model = MRICancerDetector(num_classes=2, pretrained=False)
    
    # 3. Model Training
    print("\n[4] Training Diagnostic Detector...")
    trained_model = train_mri_model(model, train_loader, epochs=3, device=device)
    
    # 4. Diagnostic Metrics
    print("\n[5] Calculating Clinical Diagnostic Metrics:")
    metrics, cm = evaluate_mri_diagnostics(trained_model, test_loader, device=device)
    for key, val in metrics.items():
        print(f"    {key}: {val}")
    print("\nConfusion Matrix:\n", cm)
    
    # 5. Grad-CAM Explainability Demo
    print("\n[6] Generating Grad-CAM Class Activation Heatmap...")
    sample_mri = data[0:1].to(device)
    sample_mri.requires_grad = True
    gradcam = GradCAM(trained_model, target_layer_name='layer4')
    heatmap, pred_class = gradcam.generate_heatmap(sample_mri)
    print(f"    Grad-CAM Heatmap Generated Successfully (Predicted Class: {'Tumor/Cancer' if pred_class == 1 else 'Normal'})")
    print("\nCancer Detection Pipeline Execution Completed Successfully!")


if __name__ == '__main__':
    main()
