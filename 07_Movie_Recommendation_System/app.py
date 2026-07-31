"""
Project 7: TMDB Movie Recommendation System - Flask Web Application
Author: Rudhra Sitholey (Reg No: 23BCY10296 | App No: IN26012560 | Email: rudhra.23bcy10296@vitbhopal.ac.in)
Port: 5000
"""

import os
from flask import Flask, render_template, request, jsonify
from src.recommender import TMDBMovieRecommender

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

recommender = TMDBMovieRecommender()


@app.route("/")
def index():
    featured_titles = [
        "Avatar",
        "Pirates of the Caribbean: At World's End",
        "The Dark Knight",
        "Inception",
        "Interstellar",
        "The Matrix",
        "Fight Club",
        "Pulp Fiction"
    ]
    return render_template(
        "index.html",
        featured_titles=featured_titles,
        student_name="Rudhra Sitholey",
        reg_no="23BCY10296",
        app_no="IN26012560",
        email="rudhra.23bcy10296@vitbhopal.ac.in"
    )


@app.route("/api/search", methods=["GET"])
def search():
    query = request.args.get("q", "").strip()
    results = recommender.search_movies(query, top_n=10)
    return jsonify({"status": "success", "results": results})


@app.route("/api/recommend", methods=["POST"])
def recommend():
    payload = request.get_json(silent=True) or {}
    title = str(payload.get("title", "Avatar")).strip()
    top_n = int(payload.get("top_n", 6))

    recommendations, query_movie = recommender.recommend(title, top_n=top_n)

    if not query_movie:
        return jsonify({
            "status": "error",
            "message": f"Movie '{title}' not found in dataset."
        }), 404

    return jsonify({
        "status": "success",
        "query_movie": query_movie,
        "recommendations": recommendations
    })


if __name__ == "__main__":
    port = 5000
    print("\n=======================================================")
    print(" Project 7: TMDB Movie Recommendation System Engine")
    print(f" Localhost Server: http://localhost:{port}")
    print("=======================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)
