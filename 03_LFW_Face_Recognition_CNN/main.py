"""
LFW Face Recognition - Main Execution Script
Author: Rudhra Sitholey (Reg No: 23BCY10296 | App No: IN26012560)
"""

import torch
from sklearn.model_selection import train_test_split
from src.embedding_extractor import FaceEmbeddingNet
from src.face_aligner import preprocess_face_images
from src.classifier import train_face_classifier, evaluate_face_recognition


def main():
    print("=" * 65)
    print(" Project 3: Face Recognition using CNN in Wild Life (LFW Dataset)")
    print(" Student: Rudhra Sitholey | Reg: 23BCY10296 | App: IN26012560")
    print("=" * 65)
    
    # 1. Load Data
    print("\n[1] Loading Preprocessed LFW Face Dataset...")
    images, labels, identity_names = preprocess_face_images(num_samples=300, num_identities=8)
    print(f"    Total Face Images: {images.size(0)} | Total Identities: {len(identity_names)}")
    
    # 2. Extract Embeddings using CNN
    print("\n[2] Extracting 128-Dimensional Face Embeddings via Deep CNN...")
    embedding_net = FaceEmbeddingNet(embedding_dim=128)
    embedding_net.eval()
    
    with torch.no_grad():
        embeddings = embedding_net(images).numpy()
    print(f"    Extracted Embeddings Shape: {embeddings.shape}")
    
    # 3. Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels.numpy(), test_size=0.25, random_state=42, stratify=labels.numpy())
    
    # 4. Train Face Matching Classifier
    print("\n[3] Training Support Vector Machine (SVM) on Face Embeddings...")
    svm_clf = train_face_classifier(X_train, y_train, method='svm')
    
    # 5. Evaluate Recognition
    print("\n[4] Evaluating Face Identification Performance...")
    acc, report = evaluate_face_recognition(svm_clf, X_test, y_test, identity_names)
    print(f"    Face Recognition Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:\n", report)
    print("LFW Face Recognition Pipeline Execution Completed Successfully!")


if __name__ == '__main__':
    main()
