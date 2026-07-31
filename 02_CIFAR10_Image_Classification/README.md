# Project 02 — CIFAR-10 Image Classification

**Author:** Rudhra Sitholey
**Registration No:** 23BCY10296 | **Application No:** IN26012560
**Email:** rudhra.23bcy10296@vitbhopal.ac.in

---

## Overview

A deep learning image classification system trained on the CIFAR-10 benchmark dataset (60,000 32×32 color images across 10 classes). The project implements a custom Convolutional Neural Network (CNN) in PyTorch, evaluates classification performance across all 10 categories, and serves real-time predictions through a Flask web application.

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Deep Learning | PyTorch, TorchVision |
| Data Processing | NumPy, Scikit-learn |
| Visualization | Matplotlib |
| Training Utilities | tqdm (progress bars) |
| Web Framework | Flask |

## Project Structure

```
02_CIFAR10_Image_Classification/
├── app.py                # Flask web server with classification endpoint
├── main.py               # CLI entry point — trains CNN on CIFAR-10
├── requirements.txt      # Python dependencies
├── src/
│   ├── dataset.py        # CIFAR-10 data loading and augmentation
│   ├── model.py          # Custom CNN architecture definition
│   └── train.py          # Training loop with loss/accuracy logging
└── templates/
    └── index.html        # Interactive classification UI
```

## How It Works

1. **Data Loading** — CIFAR-10 is downloaded via TorchVision with normalization and optional augmentation (random crops, horizontal flips).
2. **Model Architecture** — A multi-layer CNN with convolutional blocks, batch normalization, ReLU activations, max pooling, and fully connected layers classifies images into 10 categories.
3. **Training** — The network is trained using cross-entropy loss and Adam optimizer with learning rate scheduling.
4. **Inference** — The trained model predicts class probabilities for uploaded or selected sample images.

## CIFAR-10 Classes

| Index | Class |
|---|---|
| 0 | Airplane |
| 1 | Automobile |
| 2 | Bird |
| 3 | Cat |
| 4 | Deer |
| 5 | Dog |
| 6 | Frog |
| 7 | Horse |
| 8 | Ship |
| 9 | Truck |

## Getting Started

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (downloads CIFAR-10 automatically)
python main.py

# 3. Launch the web application
python app.py
# → Open http://localhost:5000
```

## Dataset

- **Source:** CIFAR-10 (Canadian Institute For Advanced Research)
- **Size:** 60,000 images (50,000 train / 10,000 test)
- **Resolution:** 32×32 RGB
- **Classes:** 10 mutually exclusive categories

