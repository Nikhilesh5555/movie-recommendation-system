
import pandas as pd

# Read the movie dataset
movies = pd.read_csv("data/movies.csv")

# Display the first 5 movies
print(movies.head())

# Count the total number of movies
print("Total movies:", len(movies))

# Display the column names
print("Columns:", movies.columns.tolist())

print("\nFirst 10 movie genres:")
print(movies["genres"].head(10))
