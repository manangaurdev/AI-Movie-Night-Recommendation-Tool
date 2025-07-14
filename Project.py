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

