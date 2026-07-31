"""
Cancer Detection using MRI - Training & Medical Diagnostic Evaluation
Author: Rudhra Sitholey (Reg No: 23BCY10296 | App No: IN26012560)
"""

import torch
import torch.nn as nn
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, confusion_matrix


def train_mri_model(model, train_loader, epochs=3, lr=1e-4, device='cpu'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch}/{epochs}] - Loss: {avg_loss:.4f}")
        
    return model


def evaluate_mri_diagnostics(model, test_loader, device='cpu'):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    sensitivity = recall_score(all_labels, all_preds)  # Sensitivity = Recall for Positive/Cancer class
    precision = precision_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    # Specificity = TN / (TN + FP)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    metrics = {
        'Accuracy': round(acc, 4),
        'Sensitivity (Recall)': round(sensitivity, 4),
        'Specificity': round(specificity, 4),
        'Precision': round(precision, 4),
        'F1-Score': round(f1, 4)
    }
    return metrics, cm
