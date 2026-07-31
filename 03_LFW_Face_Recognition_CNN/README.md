# Project 03 — LFW Face Recognition CNN

**Author:** Rudhra Sitholey
**Registration No:** 23BCY10296 | **Application No:** IN26012560
**Email:** rudhra.23bcy10296@vitbhopal.ac.in

---

## Overview

A real-time face recognition system built with PyTorch and OpenCV. The application uses a CNN-based feature extractor to generate 128-dimensional face embeddings, matches them against a registered identity database using cosine similarity, and streams live webcam video with bounding-box overlays and identity labels. New faces can be enrolled on-the-fly through a dedicated registration page.

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Deep Learning | PyTorch, TorchVision |
| Computer Vision | OpenCV (Haar cascades for detection) |
| Embeddings | 128-dim face embeddings via custom CNN |
| Similarity | Cosine similarity matching |
| Web Framework | Flask (MJPEG video streaming) |
| Serialization | Pickle (embeddings cache) |

## Project Structure

```
03_LFW_Face_Recognition_CNN/
├── app.py                # Flask server — live video stream, face registration
├── main.py               # CLI entry point — trains/evaluates face recognizer
├── recognizer.py         # Face detection, embedding extraction, matching engine
├── embeddings.pkl        # Cached face embeddings for known identities
├── requirements.txt      # Python dependencies
├── known_faces/          # Directory of registered identity images
│   └── <name>/           # One subfolder per person
├── src/
│   └── ...               # Supporting modules
└── templates/
    └── index.html        # Live webcam feed + registration UI
```

## How It Works

1. **Face Detection** — OpenCV Haar cascades detect face regions in each video frame.
2. **Embedding Extraction** — Detected faces are resized, normalized, and passed through a CNN to produce a 128-dimensional feature vector.
3. **Identity Matching** — Embeddings are compared against the registered database using cosine similarity. Matches above a threshold are labeled.
4. **Live Streaming** — Flask serves an MJPEG stream with real-time bounding boxes and identity annotations.
5. **Registration** — The `/add_face` page lets users capture a photo and register a new identity instantly.

## Getting Started

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Run the CLI evaluation
python main.py

# 3. Launch the web application
python app.py
# → Open http://localhost:5000
# → Grant camera permissions when prompted
```

## Features

- **Real-time MJPEG streaming** with face bounding boxes and identity labels
- **On-the-fly registration** — add new faces without restarting the server
- **Persistent embeddings** — cached in `embeddings.pkl` for instant startup
- **Webcam support** — works with any USB or built-in camera

## Dataset

- **Source:** Labeled Faces in the Wild (LFW) — used for training/evaluation
- **Custom Faces:** Store personal images in `known_faces/<name>/` subfolders
