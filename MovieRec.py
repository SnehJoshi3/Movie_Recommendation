import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load Data
movies = pd.read_csv('dataset.csv')
print(movies.head())
movies.info()

# 2. Create Tags
movies['tags'] = movies['genre'] + movies['overview']
print(movies.head())

# 3. Create New DataFrame
new_df = movies[['id','title','tags']]
print(new_df.head())

# 4. Vectorization
cv = CountVectorizer(max_features=5000,stop_words='english')
vec = cv.fit_transform(new_df['tags'].values.astype('U')).toarray()
print(vec.shape)

# 5. Calculate Similarity
sem = cosine_similarity(vec)

# 6. Recommendation Function
def recommend(movie_name):
    # Find the index of the movie
    try:
        index = new_df[new_df['title'] == movie_name].index[0]
        # Calculate distances
        distance = sorted(list(enumerate(sem[index])), reverse=True, key=lambda x: x[1])
        
        # Print top 5 similar movies
        print(f"\nRecommendations for '{movie_name}':")
        for i in distance[0:5]:
            print(new_df.iloc[i[0]].title)
            
    except IndexError:
        print(f"Movie '{movie_name}' not found.")

# 7. Test Calls
recommend('The Dark Knight Rises')
recommend('Dou kyu sei – Classmates')
