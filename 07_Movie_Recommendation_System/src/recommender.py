"""
TMDB Movie Recommender Engine - 4,800+ Movies Dataset with TMDB API Integration
Author: Rudhra Sitholey (Reg No: 23BCY10296 | App No: IN26012560)
"""

import os
import pickle
import requests
import pandas as pd
import numpy as np

TMDB_API_KEY = "8265bd1679663a7ea12ac168da84d2e8"
DEFAULT_POSTER = "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=500&q=80"


class TMDBMovieRecommender:
    def __init__(self, artifacts_dir=None):
        if artifacts_dir is None:
            artifacts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
        
        self.dict_path = os.path.join(artifacts_dir, "movie_dict.pkl")
        self.sim_path = os.path.join(artifacts_dir, "similarity.pkl")
        
        self.movies = None
        self.similarity = None
        self.poster_cache = {}
        self.load_models()

    def load_models(self):
        if os.path.exists(self.dict_path) and os.path.exists(self.sim_path):
            movies_dict = pickle.load(open(self.dict_path, "rb"))
            self.movies = pd.DataFrame(movies_dict)
            self.similarity = pickle.load(open(self.sim_path, "rb"))
            print(f"[OK] TMDB Movie Recommender loaded {len(self.movies)} movies and similarity matrix.")
        else:
            raise FileNotFoundError(f"Model files missing in {self.dict_path} or {self.sim_path}")

    def fetch_poster(self, movie_id):
        if movie_id in self.poster_cache:
            return self.poster_cache[movie_id]

        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                data = r.json()
                poster_path = data.get("poster_path")
                if poster_path:
                    full_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
                    self.poster_cache[movie_id] = full_url
                    return full_url
        except Exception as e:
            print(f"Poster fetch error for ID {movie_id}: {e}")

        self.poster_cache[movie_id] = DEFAULT_POSTER
        return DEFAULT_POSTER

    def search_movies(self, query, top_n=10):
        if not query or len(query.strip()) == 0:
            return self.movies['title'].head(top_n).tolist()
        
        query_str = query.strip().lower()
        matches = self.movies[self.movies['title'].str.lower().str.contains(query_str, na=False)]
        return matches['title'].head(top_n).tolist()

    def get_movie_by_title(self, title):
        matches = self.movies[self.movies['title'].str.lower() == title.strip().lower()]
        if not matches.empty:
            return matches.iloc[0].to_dict()
        return None

    def recommend(self, movie_title, top_n=6):
        movie_item = self.get_movie_by_title(movie_title)
        if not movie_item:
            # Fallback to closest match
            matches = self.movies[self.movies['title'].str.lower().str.contains(movie_title.strip().lower(), na=False)]
            if not matches.empty:
                movie_item = matches.iloc[0].to_dict()
            else:
                return [], None

        query_idx = self.movies[self.movies['movie_id'] == movie_item['movie_id']].index[0]
        distances = sorted(list(enumerate(self.similarity[query_idx])), reverse=True, key=lambda x: x[1])

        query_movie_details = {
            "movie_id": int(movie_item['movie_id']),
            "title": str(movie_item['title']),
            "year": int(movie_item['year']) if pd.notna(movie_item['year']) else 2020,
            "vote_average": round(float(movie_item['vote_average']), 1) if pd.notna(movie_item['vote_average']) else 7.0,
            "poster": self.fetch_poster(movie_item['movie_id'])
        }

        recommendations = []
        for idx, score in distances[1:top_n+1]:
            row = self.movies.iloc[idx]
            movie_id = int(row['movie_id'])
            
            recommendations.append({
                "movie_id": movie_id,
                "title": str(row['title']),
                "year": int(row['year']) if pd.notna(row['year']) else 2020,
                "vote_average": round(float(row['vote_average']), 1) if pd.notna(row['vote_average']) else 7.0,
                "similarity_score": round(float(score) * 100, 1),
                "poster": self.fetch_poster(movie_id)
            })

        return recommendations, query_movie_details
