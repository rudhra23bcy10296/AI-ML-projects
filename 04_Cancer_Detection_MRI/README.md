# Project 04 — Cancer Detection from MRI Scans

**Author:** Rudhra Sitholey
**Registration No:** 23BCY10296 | **Application No:** IN26012560
**Email:** rudhra.23bcy10296@vitbhopal.ac.in

---

## Overview

A medical diagnostic imaging application that classifies brain MRI scans as **Normal** or **Tumor/Cancer Detected** using a deep CNN built on PyTorch. The system includes Grad-CAM (Gradient-weighted Class Activation Mapping) for visual explainability — highlighting the exact regions of the MRI that drove the model's decision. Predictions are served through an interactive Flask web interface designed for clinical demonstration.

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Deep Learning | PyTorch, TorchVision |
| Explainability | Grad-CAM (custom implementation) |
| Image Processing | OpenCV, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML Utilities | Scikit-learn |
| Web Framework | Flask |

## Project Structure

```
04_Cancer_Detection_MRI/
├── app.py                      # Flask web server with MRI prediction endpoint
├── main.py                     # CLI entry point — trains and evaluates detector
├── requirements.txt            # Python dependencies
├── src/
│   ├── model.py                # MRICancerDetector CNN architecture
│   ├── image_preprocessing.py  # MRI image loading, normalization, augmentation
│   ├── gradcam.py              # Grad-CAM heatmap generation
│   └── train_eval.py           # Training loop with evaluation metrics
└── templates/
    └── index.html              # Interactive diagnostic UI with Grad-CAM overlay
```

## How It Works

1. **Image Preprocessing** — MRI scans are resized, normalized, and augmented (rotation, flipping) to improve model generalization.
2. **Model Architecture** — `MRICancerDetector` is a multi-layer CNN with convolutional blocks, batch normalization, dropout, and fully connected classification layers.
3. **Training** — The model trains on labeled MRI datasets using cross-entropy loss, tracking accuracy, precision, recall, and AUC-ROC.
4. **Grad-CAM Visualization** — Gradient-weighted activation maps highlight tumor regions, providing interpretable explanations for each prediction.
5. **Web Interface** — Users select a sample type (normal/cancer) and receive a diagnosis with confidence score and Grad-CAM heatmap overlay.

## Getting Started

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model
python main.py

# 3. Launch the diagnostic web application
python app.py
# → Open http://localhost:5000
```

## Features

- **Binary classification** — Normal vs. Tumor detection with confidence scores
- **Grad-CAM explainability** — Visual heatmaps showing which MRI regions influenced the prediction
- **Interactive web UI** — Select sample types and view real-time diagnostic results
- **Synthetic data generation** — Built-in function for creating demo MRI samples when no dataset is available

## Clinical Disclaimer

> This project is for educational and demonstration purposes only. It is **not** intended for actual medical diagnosis. Always consult qualified medical professionals for clinical decisions.
