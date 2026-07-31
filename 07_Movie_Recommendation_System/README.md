# Project 07 — TMDB Movie Recommendation System

**Author:** Rudhra Sitholey
**Registration No:** 23BCY10296 | **Application No:** IN26012560
**Email:** rudhra.23bcy10296@vitbhopal.ac.in

---

## Overview

A content-based movie recommendation engine built on the TMDB 5000 Movies dataset. The system processes movie metadata — genres, keywords, cast, crew, and plot overviews — into a unified text representation, computes TF-IDF feature vectors, and ranks movies by cosine similarity. Users interact with the system through a modern, poster-rich Flask web application with real-time search and recommendation cards.

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| NLP / Feature Extraction | Scikit-learn (TF-IDF Vectorizer) |
| Similarity Engine | SciPy (cosine distance), NumPy |
| Data Processing | Pandas |
| Web Framework | Flask |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Poster API | TMDB API (for movie poster images) |

## Project Structure

```
07_Movie_Recommendation_System/
├── app.py                      # Flask web server with search & recommendation API
├── main.py                     # CLI entry point — processes dataset, tests engine
├── requirements.txt            # Python dependencies
├── artifacts/                  # Pre-computed similarity matrices and metadata
├── src/
│   ├── recommender.py          # TMDBMovieRecommender — main recommendation class
│   ├── content_based.py        # TF-IDF + cosine similarity pipeline
│   ├── collaborative.py        # Collaborative filtering utilities
│   └── hybrid_recommender.py   # Hybrid recommendation logic
├── static/
│   └── css/style.css           # Dark-themed UI styling
└── templates/
    └── index.html              # Movie search, poster cards, recommendation UI
```

## How It Works

1. **Data Ingestion** — The TMDB 5000 dataset is loaded and cleaned. Genres, keywords, cast (top 3), director, and overview are merged into a single text blob per movie.
2. **Feature Extraction** — TF-IDF vectorization converts text blobs into sparse feature vectors (5,000-dim, English stop words removed).
3. **Similarity Computation** — Cosine similarity between all movie pairs is pre-computed and cached in `artifacts/`.
4. **Recommendation** — Given a query movie, the engine returns the top-N most similar films ranked by cosine similarity score.
5. **Web Interface** — Users search for any movie, click to select it, and instantly receive 6 visually rich recommendation cards with posters and metadata.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Main UI with featured movies |
| `GET` | `/api/search?q=<query>` | Search movies by title (fuzzy) |
| `POST` | `/api/recommend` | Get recommendations for a movie |

## Getting Started

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Run the CLI pipeline to test the engine
python main.py

# 3. Launch the web application
python app.py
# → Open http://localhost:5000
```

## Dataset

- **Source:** TMDB 5000 Movies Dataset (Kaggle)
- **Records:** 4,805 movies with metadata (genres, keywords, cast, crew, overview, ratings)
- **Features Used:** Genres, keywords, top-3 cast, director, plot overview

## Sample Recommendations

> **Query:** *The Dark Knight*
> **Results:** *The Dark Knight Rises*, *Batman Begins*, *Batman*, *Batman Returns*, *Batman Forever*, *Batman & Robin*

