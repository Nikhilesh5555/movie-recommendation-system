import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load movie data
movies = pd.read_csv("data/movies.csv")

# 2. Handle missing genres and prepare text
movies["genres"] = movies["genres"].fillna("")
movies["genres"] = movies["genres"].str.replace("|", " ", regex=False)

# 3. Convert genres into numerical features
vectorizer = TfidfVectorizer()
genre_matrix = vectorizer.fit_transform(movies["genres"])

# 4. Calculate similarity scores
similarity_matrix = cosine_similarity(genre_matrix)


# 5. Recommendation function
def recommend_movies(movie_title, number=15):

    matches = movies[
        movies["title"].str.contains(
            movie_title, case=False, regex=False, na=False
        )
    ]

    if matches.empty:
        return []

    # Select the first matching movie
    movie_index = matches.index[0]

    # Get similarity scores for the selected movie
    scores = list(enumerate(similarity_matrix[movie_index]))

    # Sort by similarity, highest first
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    recommendations = []

    for index, score in scores:
        # Do not recommend the selected movie itself
        if index == movie_index:
            continue

        recommendations.append({
            "title": movies.iloc[index]["title"],
            "similarity": round(float(score), 3)
        })

        if len(recommendations) >= number:
            break

    return recommendations


# 6. Test the function
if __name__ == "__main__":
    for movie in ["Toy Story", "Jumanji", "Grumpier Old Men"]:
        results = recommend_movies(movie, number=15)

        print(f"\nRecommendations for: {movie}")

        for item in results:
            print(item["title"], "-", item["similarity"])

        print("Total recommendations:", len(results))
