import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =====================
# 1. DATASET
# =====================
movies = pd.DataFrame({
    'title': [
        'The Dark Knight', 'Inception', 'Interstellar', 'The Matrix',
        'Avengers Endgame', 'Iron Man', 'Thor', 'Spider-Man',
        'Titanic', 'The Notebook', 'Pride and Prejudice', 'La La Land',
        'The Godfather', 'Goodfellas', 'Scarface', 'Pulp Fiction',
        'The Shining', 'Get Out', 'A Quiet Place', 'Us',
        'Toy Story', 'Finding Nemo', 'The Lion King', 'Frozen'
    ],
    'genre': [
        'Action Crime Drama', 'Action SciFi Thriller', 'SciFi Drama Adventure', 'SciFi Action',
        'Action Adventure SciFi', 'Action SciFi', 'Action Adventure Fantasy', 'Action Adventure',
        'Romance Drama', 'Romance Drama', 'Romance Drama', 'Romance Drama Music',
        'Crime Drama', 'Crime Drama', 'Crime Drama Action', 'Crime Thriller Drama',
        'Horror Thriller', 'Horror Thriller', 'Horror SciFi', 'Horror Thriller',
        'Animation Comedy', 'Animation Adventure Comedy', 'Animation Drama', 'Animation Musical'
    ],
    'rating': [9.0, 8.8, 8.6, 8.7, 8.4, 7.9, 7.9, 8.2,
               7.8, 7.9, 7.8, 8.0, 9.2, 8.7, 8.3, 8.9,
               8.4, 7.7, 7.5, 6.8, 8.3, 8.1, 8.5, 7.4],
    'description': [
        'Batman fights Joker in Gotham city dark superhero',
        'Dream heist mind bending reality layers subconscious',
        'Space time wormhole gravity relativity dimension',
        'Virtual reality simulation hacker rebels machines',
        'Superheroes unite infinity stones time travel battle',
        'Billionaire builds iron suit fights terrorists technology',
        'Norse god thunder hammer Asgard mythology',
        'Teenager superhero web city villain adventure',
        'Ship iceberg love story tragedy ocean',
        'Love story notebook summer romance couple',
        'Classic romance England society marriage prejudice',
        'Jazz music Los Angeles dreams ambition love',
        'Mafia family power loyalty betrayal crime',
        'Gangster mob crime street loyalty violence',
        'Drug dealer crime Miami power cocaine',
        'Crime stories hitman redemption nonlinear',
        'Hotel haunted isolation cabin horror writer',
        'Sunken place racism thriller social horror',
        'Monsters silence family survival horror',
        'Doppelgangers family shadow copies horror',
        'Toys alive friendship cowboy spaceman adventure',
        'Fish ocean father son clownfish coral reef',
        'Lion cub king pride lands circle life',
        'Ice queen sister magic kingdom snow'
    ]
})

print("Dataset loaded!")
print(f"Total movies: {len(movies)}")

# =====================
# 2. TF-IDF MODEL
# =====================
movies['combined'] = movies['genre'] + ' ' + movies['description']

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(movies['combined'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print("Similarity matrix ready!")

# =====================
# 3. RECOMMEND FUNCTION
# =====================
def recommend(movie_title, n=5):
    # Movie dhundho
    matches = movies[movies['title'].str.lower() == movie_title.lower()]
    
    if matches.empty:
        # Partial match try karo
        matches = movies[movies['title'].str.lower().str.contains(movie_title.lower())]
        if matches.empty:
            print(f"\nMovie '{movie_title}' nahi mili!")
            print("Available movies:")
            for t in movies['title']:
                print(f"  - {t}")
            return
    
    idx = matches.index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    
    print(f"\n{'='*50}")
    print(f"Movies similar to: {movies['title'][idx]}")
    print(f"{'='*50}")
    
    for i, (movie_idx, score) in enumerate(sim_scores, 1):
        print(f"{i}. {movies['title'][movie_idx]}")
        print(f"   Genre: {movies['genre'][movie_idx]}")
        print(f"   Rating: {movies['rating'][movie_idx]}/10")
        print(f"   Similarity: {score*100:.1f}%")
        print()

# =====================
# 4. VISUALIZATION
# =====================
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Movie Recommender — Analysis', fontsize=16, fontweight='bold')

# Graph 1 - Genre Distribution
genre_counts = {}
for genres in movies['genre']:
    for g in genres.split():
        genre_counts[g] = genre_counts.get(g, 0) + 1
genre_df = pd.Series(genre_counts).sort_values(ascending=False).head(8)
genre_df.plot(kind='bar', ax=axes[0], color='steelblue', edgecolor='black')
axes[0].set_title('Top Genres')
axes[0].set_xlabel('Genre')
axes[0].set_ylabel('Count')
axes[0].tick_params(rotation=45)

# Graph 2 - Rating Distribution
axes[1].hist(movies['rating'], bins=10, color='coral', edgecolor='black')
axes[1].set_title('Rating Distribution')
axes[1].set_xlabel('Rating')
axes[1].set_ylabel('Count')

# Graph 3 - Similarity Heatmap (first 10)
sns.heatmap(cosine_sim[:10, :10], 
            xticklabels=movies['title'][:10].str[:10],
            yticklabels=movies['title'][:10].str[:10],
            cmap='YlOrRd', ax=axes[2], annot=True, fmt='.1f')
axes[2].set_title('Movie Similarity Matrix')
axes[2].tick_params(axis='x', rotation=45)
axes[2].tick_params(axis='y', rotation=0)

plt.tight_layout()
plt.savefig('movie_analysis.png')
plt.show()
print("Graph saved!")

# =====================
# 5. INTERACTIVE
# =====================
print("\n" + "="*50)
print("MOVIE RECOMMENDER SYSTEM")
print("="*50)

while True:
    movie = input("\nKaunsi movie jaisi recommend chahiye? (q = quit): ")
    if movie.lower() == 'q':
        break
    recommend(movie)

print("\nProgram finished!")