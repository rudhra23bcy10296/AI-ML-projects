"""
CIFAR-10 Training & Evaluation Module
Author: Rudhra Sitholey (Reg No: 23BCY10296 | App No: IN26012560)
"""

import torch
import torch.nn as nn
import torch.optim as optim


def train_cifar_model(model, train_loader, test_loader, epochs=5, lr=0.001, device='cpu'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print(f"Starting model training on device: {device}")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        scheduler.step()
        train_acc = 100.0 * correct / total
        epoch_loss = running_loss / total
        
        # Test evaluation
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
        print(f"Epoch [{epoch}/{epochs}] - Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
        
    return model


def evaluate_model(model, test_loader, criterion=nn.CrossEntropyLoss(), device='cpu'):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return running_loss / total, 100.0 * correct / total
