"""
Movie Recommendation System - Hybrid Recommender System Engine
Author: Rudhra Sitholey (Reg No: 23BCY10296 | App No: IN26012560)
"""

import pandas as pd
from src.content_based import ContentBasedRecommender
from src.collaborative import CollaborativeFilteringSVD


def get_sample_movie_data():
    movies = pd.DataFrame([
        {
            'movie_id': 1,
            'title': 'The Matrix',
            'genres': 'Action Sci-Fi',
            'overview': 'A computer hacker learns about the true nature of reality and his role in the war against its controllers.',
            'year': 1999,
            'poster': 'https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5?w=400&q=80'
        },
        {
            'movie_id': 2,
            'title': 'Inception',
            'genres': 'Action Sci-Fi Mystery',
            'overview': 'A thief who steals corporate secrets through dream-sharing technology is given the inverse task of planting an idea.',
            'year': 2010,
            'poster': 'https://images.unsplash.com/photo-1534447677768-be436bb09401?w=400&q=80'
        },
        {
            'movie_id': 3,
            'title': 'Interstellar',
            'genres': 'Adventure Drama Sci-Fi',
            'overview': 'A team of explorers travel through a wormhole in space in an attempt to ensure humanity survival.',
            'year': 2014,
            'poster': 'https://images.unsplash.com/photo-1451187580459-43490279c0fa?w=400&q=80'
        },
        {
            'movie_id': 4,
            'title': 'The Dark Knight',
            'genres': 'Action Crime Drama',
            'overview': 'When the menace known as the Joker wreaks havoc and chaos on Gotham, Batman must accept one of the greatest tests.',
            'year': 2008,
            'poster': 'https://images.unsplash.com/photo-1509198397868-475647b2a1e5?w=400&q=80'
        },
        {
            'movie_id': 5,
            'title': 'Pulp Fiction',
            'genres': 'Crime Drama',
            'overview': 'The lives of two mob hitmen, a boxer, a gangster and his wife intertwine in four tales of violence and redemption.',
            'year': 1994,
            'poster': 'https://images.unsplash.com/photo-1594909122845-11baa439b7bf?w=400&q=80'
        },
        {
            'movie_id': 6,
            'title': 'Avatar',
            'genres': 'Action Adventure Sci-Fi',
            'overview': 'A paraplegic Marine dispatched to the moon Pandora on a unique mission becomes torn between following orders.',
            'year': 2009,
            'poster': 'https://images.unsplash.com/photo-1518709268805-4e9042af9f23?w=400&q=80'
        },
        {
            'movie_id': 7,
            'title': 'Fight Club',
            'genres': 'Drama Thriller',
            'overview': 'An insomniac office worker and a devil-may-care soap maker form an underground fight club that evolves.',
            'year': 1999,
            'poster': 'https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?w=400&q=80'
        },
        {
            'movie_id': 8,
            'title': 'The Shawshank Redemption',
            'genres': 'Drama Crime',
            'overview': 'Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.',
            'year': 1994,
            'poster': 'https://images.unsplash.com/photo-1507679799987-c73779587ccf?w=400&q=80'
        },
        {
            'movie_id': 9,
            'title': 'Dune',
            'genres': 'Action Adventure Sci-Fi',
            'overview': 'A noble family becomes embroiled in a war for control over the galaxy most valuable asset while its heir is troubled by visions.',
            'year': 2021,
            'poster': 'https://images.unsplash.com/photo-1509198397868-475647b2a1e5?w=400&q=80'
        },
        {
            'movie_id': 10,
            'title': 'Oppenheimer',
            'genres': 'Biography Drama History',
            'overview': 'The story of American scientist J. Robert Oppenheimer and his role in the development of the atomic bomb.',
            'year': 2023,
            'poster': 'https://images.unsplash.com/photo-1446776811953-b23d57bd21aa?w=400&q=80'
        }
    ])
    
    # User interaction matrix
    ratings = pd.DataFrame([
        {'user_id': 101, 'movie_id': 1, 'rating': 5.0},
        {'user_id': 101, 'movie_id': 2, 'rating': 4.5},
        {'user_id': 101, 'movie_id': 3, 'rating': 5.0},
        {'user_id': 101, 'movie_id': 9, 'rating': 4.8},
        {'user_id': 102, 'movie_id': 4, 'rating': 5.0},
        {'user_id': 102, 'movie_id': 5, 'rating': 4.0},
        {'user_id': 102, 'movie_id': 8, 'rating': 4.9},
        {'user_id': 103, 'movie_id': 1, 'rating': 4.0},
        {'user_id': 103, 'movie_id': 6, 'rating': 5.0},
        {'user_id': 103, 'movie_id': 10, 'rating': 4.7},
    ])
    return movies, ratings


class HybridRecommender:
    def __init__(self):
        self.content_engine = ContentBasedRecommender()
        self.collab_engine = CollaborativeFilteringSVD()

    def fit(self, movies_df, ratings_df):
        self.content_engine.fit(movies_df)
        self.collab_engine.fit(ratings_df)

    def get_recommendations(self, movie_title, user_id=101, top_n=4):
        content_recs = self.content_engine.recommend(movie_title, top_n=top_n)
        collab_movie_ids = self.collab_engine.predict_user_recommendations(user_id, top_n=top_n)
        return content_recs, collab_movie_ids
