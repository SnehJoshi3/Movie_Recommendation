import sqlite3
database = sqlite3.connect("contributions.db")
cursor = database.cursor()
conn = sqlite3.connect("contributions.db")
cursor = conn.cursor()
query = "create table contributions (movie_name varchar(100), tmdb_or_imdb_link varchar(100));"
cursor.execute(query)
database.commit()
database.close()
