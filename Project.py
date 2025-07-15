"""
Build an AI Movie Night Recommendation Tool

To do this, we will be using two data sets: 
- Movies metadata: A data set containing metadata of about 9000 movies (title, description, etc.)
- User ratings: A data set containing ratings of how much someone liked a movie. 

We will be building towards our end goal by covering the following tasks: 
- Understanding the data set by doing some basic exploratory analysis 
- Building a first recommender based on movie popularity or movie ratings 
- Personalising recommendations by exploiting user ratings 
- Leveraging LLMs to calculate similarity between movies 
- Generating a recommendation by writing what kind of movies you'd like to see 
- Putting it all together into one single recommendation tool


The data is contained in two CSV files named `movies_metadata.csv` and `ratings.csv`

`movies_metadata` contains the following columns: 

- `movie_id`: Unique identifier of each movie. 
- `title`: Title of the movie. 
- `overview`: Short description of the movie. 
- `vote_average`: Average score the movie got.
- `vote_count`: Total number of votes the movie got. 

`ratings` contains the following columns: 

- `user_id`: Unique identifier of the person who rated the movie. 
- `movie_id`: Unique identifier of the movie. 
- `rating`: Value between 0 and 10 indicating how much the person liked the movie. 

"""

# Task 1: Import the ratings and movie metadata and explore it. 
import pandas as pd
import matplotlib.pyplot as plt

# Reading the movies_metadata file
movies_metadata = pd.read_csv('movies_metadata.csv')

# Reading the ratings file
ratings = pd.read_csv('ratings.csv')

# Counting how many unique movies there are
unique_movies = movies_metadata['movie_id'].nunique()

# Counting how many unique users have rated how many unique movies
unique_users = ratings['user_id'].nunique()
unique_rated_movies = ratings['movie_id'].nunique()

print(f"Number of unique movies: {unique_movies}")
print(f"Number of unique users: {unique_users}")

# Visualse the vote_average column
plt.figure(figsize=(10,5))
plt.hist(movies_metadata['vote_average'].dropna(), bins=20, edgecolor='k', alpha=0.7)
plt.title('Distribution of Vote Average')
plt.xlabel('Vote Count')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Task-2: Simple recommender based on popularity or highest rating

"""
In short, a recommender is any system that generates suggestions for an end user. We will start with creating the simplest recommender, one that ranks all movies according to the highest average score, or the highest number of votes. 

This kind of recommender generates the same output for anyone using it.

"""
def simple_recommender(movies_metadata, criterion='vote_average', top_n=10):
    if criterion not in ['vote_average', 'vote_count']:
        raise ValueError("Criterion must be either 'vote_average' or 'vote_count'")
    
    # Sort the movies based on the specified criterion
    recommended_movies = movies_metadata.sort_values(by=criterion, ascending=False)

    # Select the top N movies
    recommended_movies = recommended_movies[['movie_id', 'title', 'overview', criterion]].head(top_n)

    return recommended_movies

top_movies_by_average = simple_recommender(movies_metadata, criterion='vote_average', top_n=10)
top_movies_by_count = simple_recommender(movies_metadata, criterion='vote_count', top_n=10)

# Display the top movies
top_movies_by_average
# top_movies_by_count


# Task-3: Generate recommendations based on user ratings


from sklearn.metrics.pairwise import cosine_similarity

def create_user_based_recommender(movies_metadata, ratings, movie_title, top_n=10):
    # Merge movies_metadata with ratings
    movie_ratings = pd.merge(ratings, movies_metadata, on='movie_id')

    # Create a pivot table with users as rows, movies as columns, and ratings as values
    user_movie_matrix = movie_ratings.pivot_table(index='user_id', columns='title', values='rating')

    # Fill NaN values with 0 (assuming unrated movies have a rating of 0)
    user_movie_matrix.fillna(0, inplace=True)

    # Compute the cosine similarity matrix
    movie_similarity = cosine_similarity(user_movie_matrix.T)

    # Covert the similarity matrix to a DataFrame
    movie_similarity_df = pd.DataFrame(movie_similarity, index=user_movie_matrix.columns, columns=user_movie_matrix.columns)

    # Get the list of similar movies
    similar_movies = movie_similarity_df[movie_title].sort_values(ascending=False)[1:top_n+1]

    return similar_movies

movie_title = "The Godfather" # Replace with the movie title you want to get recommendations for
recommended_movies = create_user_based_recommender(movies_metadata, ratings, movie_title, top_n=10)
# recommended_movies 

# Task-4:- Generate embeddings based on the movie descriptions

from sentence_transformers import SentenceTransformer
import pandas as pd
from tqdm import tqdm

# Load a pre-trained model from Sentence Transformers
model = SentenceTransformer('all-MiniLM-L6-v2')

# Ensure the 'overview' column is filled with strings
movies_metadata['overview'] = movies_metadata['overview'].fillna('').astype(str)

# Generate embeddings for each movie overview
tqdm.pandas(desc="Generating embeddings")
movies_metadata['embedding'] = movies_metadata['overview'].progress_apply(lambda x: model.encode(x).tolist())


# Task-5:- Use embedding similarity to generate recommendations
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def recommend_movies(description, n=5):
    # Generate embedding for the user input description
    user_embedding = model.encode(description).tolist()

    # Calculate cosine similarity between user embedding and all movie embeddings
    movies_metadata['similarity'] = movies_metadata['embedding'].apply(lambda x: cosine_similarity([user_embedding]))

    # Sort movies by similarity and get top n recommendations
    recommendations = movies_metadata.sort_values(by='similarity', ascending=False).head(n)

    return recommendations[['title', 'overview', 'similarity']]

# Example usage
user_description = "A thrilling adventure with lots of action and suspense."
recommendations = recommend_movies(user_description, n=5)
recommendations