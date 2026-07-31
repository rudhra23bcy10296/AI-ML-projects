"""
CIFAR-10 Image Classification - Main Execution Script
Author: Rudhra Sitholey (Reg No: 23BCY10296 | App No: IN26012560)
"""

import torch
from src.model import ResNetCIFAR
from src.dataset import get_dataloaders
from src.train import train_cifar_model, evaluate_model


def main():
    print("=" * 65)
    print(" Project 2: CIFAR-10 Image Classification using CNN")
    print(" Student: Rudhra Sitholey | Reg: 23BCY10296 | App: IN26012560")
    print("=" * 65)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[1] Hardware Acceleration: {device}")
    
    print("\n[2] Initializing DataLoaders...")
    train_loader, test_loader, classes = get_dataloaders(batch_size=32)
    print(f"    CIFAR-10 Classes: {classes}")
    
    print("\n[3] Building Deep ResNet CIFAR Architecture...")
    model = ResNetCIFAR(num_classes=10)
    
    print("\n[4] Training Deep CNN Model...")
    trained_model = train_cifar_model(model, train_loader, test_loader, epochs=3, device=device)
    
    print("\n[5] Final Evaluation:")
    _, final_acc = evaluate_model(trained_model, test_loader, device=device)
    print(f"    Final Model Accuracy: {final_acc:.2f}%")
    print("\nCIFAR-10 Classification Task Completed Successfully!")


if __name__ == '__main__':
    main()
