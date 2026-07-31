"""
Movie Recommendation System - Content-Based Filtering Module
Author: Rudhra Sitholey (Reg No: 23BCY10296 | App No: IN26012560)
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
import numpy as np


class ContentBasedRecommender:
    """
    Content-Based filtering using TF-IDF metadata vectorization and Cosine Similarity.
    """
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        self.movies_df = None

    def fit(self, movies_df):
        self.movies_df = movies_df.reset_index(drop=True)
        # Combine genres and plot overview metadata
        self.movies_df['metadata'] = self.movies_df['genres'] + " " + self.movies_df['overview']
        self.tfidf_matrix = self.vectorizer.fit_transform(self.movies_df['metadata'])

    def recommend(self, movie_title, top_n=5):
        idx_matches = self.movies_df[self.movies_df['title'].str.lower() == movie_title.lower()].index
        if len(idx_matches) == 0:
            return pd.DataFrame()
            
        idx = idx_matches[0]
        cosine_sim = linear_kernel(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        sim_scores = list(enumerate(cosine_sim))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        
        movie_indices = [i[0] for i in sim_scores]
        results = self.movies_df.iloc[movie_indices].copy()
        results['similarity_score'] = [round(i[1], 4) for i in sim_scores]
        return results[['title', 'genres', 'similarity_score']]
