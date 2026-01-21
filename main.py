# Movie Recommendation System (Machine Learning Project)
# This project recommends movies based on content similarity

import numpy as np
import pandas as pd
import ast
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Load movie and credits datasets
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# Merge both datasets using movie title
movies = movies.merge(credits, on='title')

# Select only useful columns

# | Column   | Meaning                         |
# | -------- | ------------------------------- |
# | movie_id | Unique ID of the movie          |
# | title    | Name of the movie               |
# | overview | Short story or description      |
# | genres   | Category or type of the movie   |
# | cast     | Main actors in the movie        |
# | crew     | Director of the movie           |

movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]

# Convert genre and keyword data from string to list
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

# Get only top 3 actors from cast
def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
            counter += 1
    return L

movies['cast'] = movies['cast'].apply(convert3)

# Extract director name from crew data
def fetch_director(text):
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            return [i['name']]
    return []

movies['crew'] = movies['crew'].apply(fetch_director)

# Fill missing overview values and split text into words
movies['overview'] = movies['overview'].fillna('').apply(lambda x: x.split())

# Remove spaces from words to make tags cleaner
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ","") for i in x])

# Combine all important information into one column
movies['tags'] = (
    movies['overview'] +
    movies['genres'] +
    movies['keywords'] +
    movies['cast'] +
    movies['crew']
)

# Create a new DataFrame with only required columns
new_df = movies[['movie_id','title','tags']].copy()

# Convert list of tags into a single string
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

# Convert text data into numerical vectors
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

# Calculate similarity between movies
similarity = cosine_similarity(vectors)

# Function to recommend movies
def recommend(movie):
    # Find index of the given movie
    index = new_df[new_df['title'] == movie].index[0]

    # Sort movies based on similarity score
    distances = sorted(
        list(enumerate(similarity[index])),
        reverse=True,
        key=lambda x: x[1]
    )

    # Print top 5 similar movies
    for i in distances[1:6]:
        print(new_df.iloc[i[0]].title)

# Test the recommendation system
recommend('Avatar')