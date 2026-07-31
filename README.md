# AI / ML Project Portfolio

**Author:** Rudhra Sitholey
**Registration No:** 23BCY10296 | **Application No:** IN26012560
**Email:** rudhra.23bcy10296@vitbhopal.ac.in

---

## Overview

A comprehensive portfolio of **7 interactive AI/ML web applications** spanning Supervised Learning, Deep Learning, Computer Vision, Medical Imaging, Recommender Systems, Full-Stack Deployment, and Generative AI. Each project is self-contained with its own dataset, model pipeline, web interface, and documentation.

---

## Projects

| # | Project | Domain | Key Tech | Port |
|---|---|---|---|---|
| 01 | [Adult Census Income Classification](./01_Adult_Census_Income_Classification/) | Supervised Learning | Scikit-learn, XGBoost, LightGBM | 5000 |
| 02 | [CIFAR-10 Image Classification](./02_CIFAR10_Image_Classification/) | Deep Learning / CV | PyTorch CNN | 5000 |
| 03 | [LFW Face Recognition CNN](./03_LFW_Face_Recognition_CNN/) | Computer Vision | PyTorch, OpenCV, Webcam | 5000 |
| 04 | [Cancer Detection from MRI](./04_Cancer_Detection_MRI/) | Medical Imaging | PyTorch, Grad-CAM | 5000 |
| 07 | [TMDB Movie Recommendation System](./07_Movie_Recommendation_System/) | Recommender Systems | TF-IDF, Cosine Similarity | 5000 |
| 08 | [End-to-End House Price Prediction](./08_End_to_End_Render_Deployment/) | Full-Stack Deployment | Flask + React + Docker + Render | 5000 |
| 09 | [RAG Chatbot Capstone](./09_RAG_Chatbot_Capstone/) | Generative AI / NLP | Groq LLM, ChromaDB, FastAPI | 8000 |

---

## Project Summaries

### 01 — Adult Census Income Classification
Predicts whether an individual's income exceeds $50K/year using the UCI Adult Census dataset. Compares Logistic Regression, Random Forest, XGBoost, and LightGBM. Includes a Flask web UI for interactive predictions.

### 02 — CIFAR-10 Image Classification
Classifies 32×32 color images into 10 categories (airplane, automobile, bird, etc.) using a custom PyTorch CNN trained on the CIFAR-10 benchmark dataset.

### 03 — LFW Face Recognition CNN
Real-time face recognition with live webcam streaming. Uses CNN-generated 128-dim embeddings and cosine similarity matching. Supports on-the-fly identity registration.

### 04 — Cancer Detection from MRI
Classifies brain MRI scans as Normal or Tumor using a PyTorch CNN. Includes Grad-CAM visualizations for interpretable, explainable AI — highlighting regions that drove the diagnosis.

### 07 — TMDB Movie Recommendation System
Content-based recommendation engine built on the TMDB 5000 Movies dataset. Processes genres, keywords, cast, crew, and overviews into TF-IDF vectors and ranks movies by cosine similarity. Features a poster-rich search and recommendation interface.

### 08 — End-to-End House Price Prediction (Render Deployment)
Full-stack ML application with a Flask API backend, React + Vite frontend, and Docker containerization. Predicts King County house prices using Linear Regression. Configured for one-click deployment on Render.

### 09 — RAG Chatbot Capstone
Retrieval-Augmented Generation chatbot that answers questions from custom documents. Ingests files into ChromaDB, retrieves relevant context via semantic search, and generates grounded answers using Groq-hosted LLMs (LLaMA 3 / Mixtral).

---

## Quick Start

### Prerequisites

- Python 3.10+
- pip
- Node.js 18+ (for Project 08 frontend only)
- Webcam (for Project 03 only)
- Groq API key (for Project 09 only)

### Running Any Project

```bash
# Navigate to the project directory
cd <project_folder>

# Install Python dependencies
pip install -r requirements.txt

# Launch the application
python app.py        # Projects 01–04, 07, 08
# or
python main.py       # Project 09 (FastAPI)
```

### Project-Specific Setup

| Project | Extra Steps |
|---|---|
| **08** | `cd frontend && npm install && npm run build && cd ..` before running `python app.py` |
| **09** | Copy `.env.example` → `.env` and add your `GROQ_API_KEY`. Run `python ingest.py` to populate the vector store. |

---

## Tech Stack Overview

```
Supervised ML       → Scikit-learn, XGBoost, LightGBM
Deep Learning       → PyTorch, TorchVision
Computer Vision     → OpenCV, CNN Embeddings
Explainability      → Grad-CAM
NLP / Text          → TF-IDF, Cosine Similarity
Vector Search       → ChromaDB
LLM Integration     → Groq Cloud (LLaMA 3, Mixtral)
Web Backends        → Flask, FastAPI
Web Frontends       → Jinja2 Templates, React + Vite, Vanilla JS
Containerization    → Docker
Cloud Deployment    → Render
```

---

## Repository Structure

```
ALL Project/
├── README.md                                 ← You are here
├── 01_Adult_Census_Income_Classification/
├── 02_CIFAR10_Image_Classification/
├── 03_LFW_Face_Recognition_CNN/
├── 04_Cancer_Detection_MRI/
├── 07_Movie_Recommendation_System/
├── 08_End_to_End_Render_Deployment/
└── 09_RAG_Chatbot_Capstone/
```

Each project folder contains its own `README.md` with detailed documentation, setup instructions, and technical architecture.

---

## License

This repository is for educational and portfolio demonstration purposes.
