import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import sqlite3

conn = sqlite3.connect('movies_database.db')
movies = pd.read_sql_query("SELECT * FROM movies", conn)
conn.close()


required_columns = ['title', 'overview', 'genres', 'cast', 'director', 'poster_path', 'vote_average', 'vote_count','original_language','popularity']
movies = movies[required_columns].dropna()
movies['tag'] = movies['overview'] + ' ' + movies['genres'] + ' ' + movies['cast'] + ' ' + movies['director']
movies = movies.sort_values(by="vote_count", ascending=False).head(10000).reset_index(drop=True)


tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
vectorized_data = tfidf.fit_transform(movies['tag'])

pickle.dump(movies.to_dict('records'), open('movies_dict.pkl', 'wb'))
pickle.dump(vectorized_data, open('vectorized_data.pkl', 'wb'))

print("Model successfully trained and pickled!")