"""
Movie Recommendation System - Main Execution Script
Author: Rudhra Sitholey (Reg No: 23BCY10296 | App No: IN26012560)
"""

from src.hybrid_recommender import HybridRecommender, get_sample_movie_data


def main():
    print("=" * 65)
    print(" Project 7: Movie Recommendation System")
    print(" Student: Rudhra Sitholey | Reg: 23BCY10296 | App: IN26012560")
    print("=" * 65)
    
    print("\n[1] Loading Movie Metadata & User Rating Matrices...")
    movies_df, ratings_df = get_sample_movie_data()
    print(f"    Loaded {len(movies_df)} Movies | {len(ratings_df)} User Rating Logs")
    
    print("\n[2] Training Hybrid Recommender Engine...")
    recommender = HybridRecommender()
    recommender.fit(movies_df, ratings_df)
    
    query_movie = "The Matrix"
    user_id = 101
    print(f"\n[3] Generating Content-Based Recommendations for: '{query_movie}'...")
    content_recs, collab_ids = recommender.get_recommendations(query_movie, user_id=user_id, top_n=3)
    print(content_recs.to_string(index=False))
    
    print(f"\n[4] Collaborative SVD Movie Recommendations for User ID {user_id}:")
    rec_movies = movies_df[movies_df['movie_id'].isin(collab_ids)]['title'].tolist()
    print("    Recommended Titles:", rec_movies)
    print("\nMovie Recommendation Pipeline Execution Completed Successfully!")


if __name__ == '__main__':
    main()
