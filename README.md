# AI / ML Project Portfolio

**Author:** Rudhra Sitholey
**Registration No:** 23BCY10296 | **Application No:** IN26012560
**Email:** rudhra.23bcy10296@vitbhopal.ac.in

---

## Overview

A comprehensive portfolio of **9 interactive AI/ML projects** spanning Supervised Learning, Deep Learning, Computer Vision, Medical Imaging, Reinforcement Learning, Recommender Systems, Full-Stack Deployment, and Generative AI. Each project is self-contained with its own dataset, model pipeline, visual simulation, and documentation.

---

## Projects

| # | Project | Domain | Key Tech | Type / Output |
|---|---|---|---|---|
| 01 | [Adult Census Income Classification](./01_Adult_Census_Income_Classification/) | Supervised Learning | Scikit-learn, XGBoost, LightGBM | Web App (Port 5000) |
| 02 | [CIFAR-10 Image Classification](./02_CIFAR10_Image_Classification/) | Deep Learning / CV | PyTorch CNN | Web App (Port 5000) |
| 03 | [LFW Face Recognition CNN](./03_LFW_Face_Recognition_CNN/) | Computer Vision | PyTorch, OpenCV, Webcam | Real-time Web App |
| 04 | [Cancer Detection from MRI](./04_Cancer_Detection_MRI/) | Medical Imaging | PyTorch, Grad-CAM | Web App (Port 5000) |
| 05 | [CartPole Reinforcement Learning](./05_CartPole_RL_Agent/) | Reinforcement Learning | PyTorch, Gymnasium, DQN | Pygame Visual Simulation |
| 06 | [Lunar Lander Reinforcement Learning](./06_Lunar_Lander_RL_Agent/) | Reinforcement Learning | PyTorch, Gymnasium Box2D, Dueling DQN | Custom Pygame Visual Simulation |
| 07 | [TMDB Movie Recommendation System](./07_Movie_Recommendation_System/) | Recommender Systems | TF-IDF, Cosine Similarity | Web App (Port 5000) |
| 08 | [End-to-End House Price Prediction](./08_End_to_End_Render_Deployment/) | Full-Stack Deployment | Flask + React + Docker + Render | Web App ([Live Demo](https://house-price-prediction-webapp-b4jv.onrender.com/)) |
| 09 | [RAG Chatbot Capstone](./09_RAG_Chatbot_Capstone/) | Generative AI / NLP | Groq LLM, ChromaDB, FastAPI | Web App (Port 8000) |

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

### 05 — CartPole Reinforcement Learning Agent
Solves the `CartPole-v1` control environment using a PyTorch Deep Q-Network (`DQN`) with an Experience Replay Buffer and Epsilon-Greedy policy. Features a live native Pygame window runner (`visualize.py`).

### 06 — Lunar Lander Reinforcement Learning Agent
Solves the `LunarLander` Box2D control environment using a Dueling Deep Q-Network (`DuelingDQN`) with Double Q-Learning and Soft Target Updates ($\tau = 0.001$). Includes a custom styled Pygame simulation window featuring a **flat moon surface**, **2 blue landing pad markers**, **white spaceship**, and **red engine thruster sparks**.

### 07 — TMDB Movie Recommendation System
Content-based recommendation engine built on the TMDB 5000 Movies dataset. Processes genres, keywords, cast, crew, and overviews into TF-IDF vectors and ranks movies by cosine similarity. Features a poster-rich search and recommendation interface.

### 08 — End-to-End House Price Prediction (Render Deployment)
Full-stack ML application with a Flask API backend, React + Vite frontend, and Docker containerization. Predicts King County house prices using Linear Regression. Configured for one-click deployment on Render.  
**Live Demo:** [https://house-price-prediction-webapp-b4jv.onrender.com/](https://house-price-prediction-webapp-b4jv.onrender.com/)

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

# Launch Web Application (Projects 01–04, 07, 08)
python app.py

# Launch Visual RL Window Simulation (Projects 05, 06)
python visualize.py
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
├── 05_CartPole_RL_Agent/
├── 06_Lunar_Lander_RL_Agent/
├── 07_Movie_Recommendation_System/
├── 08_End_to_End_Render_Deployment/
└── 09_RAG_Chatbot_Capstone/
```

Each project folder contains its own `README.md` with detailed documentation, setup instructions, and technical architecture.

---

## License

This repository is for educational and portfolio demonstration purposes.

