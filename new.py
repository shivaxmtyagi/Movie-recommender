import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Set up the web page
st.set_page_config(page_title="Movie Recommender", page_icon="🍿")
st.title("Movie Recommender🍿🍿")
st.write("This is a content-based recommendation system based on genres and directors.")
st.markdown("---")

# 2. The Dataset 
@st.cache_data
def load_data():
    data = {
        'movie_id': [1, 2, 3, 4, 5],
        'title': ['The Matrix', 'Inception', 'Interstellar', 'The Godfather', 'Pulp Fiction'],
        'genre': ['Action Sci-Fi', 'Action Sci-Fi', 'Adventure Sci-Fi', 'Crime Drama', 'Crime Drama'],
        'director': ['Wachowskis', 'Christopher Nolan', 'Christopher Nolan', 'Francis Ford Coppola', 'Quentin Tarantino']
    }
    return pd.DataFrame(data)

df = load_data()

# 3. Model Preparation
@st.cache_data
def prepare_model(dataframe):
    dataframe['combined_features'] = dataframe['genre'] + " " + dataframe['director']
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(dataframe['combined_features'])
    similarity = cosine_similarity(feature_vectors)
    return similarity

similarity_matrix = prepare_model(df)

# 4. Frontend UI - User Input
st.subheader("Find a Movie")
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown:",
    df['title'].values
)

# 5. Execute Recommendation on Button Click
if st.button("Get Recommendations"):
    # Find the index
    movie_index = df[df['title'] == selected_movie].index[0]
    
    # Calculate similarity
    similarity_scores = list(enumerate(similarity_matrix[movie_index]))
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Display Results
    st.success(f"Because you watched **{selected_movie}**, you might like: Oppenheimer")
    
    for i, score in sorted_scores[1:4]:
        recommended_movie = df.iloc[i]['title']
        match_percentage = round(score * 100, 2)