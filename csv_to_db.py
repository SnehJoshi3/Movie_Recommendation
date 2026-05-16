import pandas as pd
import sqlite3

movies_df = pd.read_csv('TMDB_all_movies.csv')
conn = sqlite3.connect('movies_database.db')
movies_df.to_sql('movies', conn, if_exists='replace', index=False)
conn.close()

print("CSV successfully transferred to SQLite Database!")