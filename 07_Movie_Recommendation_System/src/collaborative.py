"""
Movie Recommendation System - Collaborative Filtering (SVD Matrix Factorization)
Author: Rudhra Sitholey (Reg No: 23BCY10296 | App No: IN26012560)
"""

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds


class CollaborativeFilteringSVD:
    """
    Collaborative filtering via Singular Value Decomposition (SVD) Matrix Factorization.
    """
    def __init__(self, k_factors=10):
        self.k = k_factors
        self.user_predicted_ratings = None
        self.user_movie_pivot = None

    def fit(self, ratings_df):
        self.user_movie_pivot = ratings_df.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
        ratings_matrix = self.user_movie_pivot.values
        
        # De-mean the data (normalize by user mean)
        user_ratings_mean = np.mean(ratings_matrix, axis=1)
        ratings_demeaned = ratings_matrix - user_ratings_mean.reshape(-1, 1)
        
        # Singular Value Decomposition
        U, sigma, Vt = svds(ratings_demeaned, k=min(self.k, min(ratings_matrix.shape) - 1))
        sigma = np.diag(sigma)
        
        predicted = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
        self.user_predicted_ratings = pd.DataFrame(predicted, index=self.user_movie_pivot.index, columns=self.user_movie_pivot.columns)

    def predict_user_recommendations(self, user_id, top_n=5):
        if user_id not in self.user_predicted_ratings.index:
            return []
            
        user_preds = self.user_predicted_ratings.loc[user_id].sort_values(ascending=False)
        return user_preds.head(top_n).index.tolist()
