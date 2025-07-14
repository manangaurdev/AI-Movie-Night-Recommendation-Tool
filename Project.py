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