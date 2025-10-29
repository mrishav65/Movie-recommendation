import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Load dataset
# -----------------------------
movies = pd.read_csv("movies.csv")

# Clean column names for safety
movies.columns = movies.columns.str.strip()
movies['Genre'] = movies['Genre'].astype(str)
movies['Title'] = movies['Title'].astype(str)

# Combine genres + titles for better matching
movies['Combined'] = movies['Title'] + " " + movies['Genre']

# TF-IDF Vectorizer to understand similarity between movies
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(movies['Combined'])

# -----------------------------
# Function: Recommend movies
# -----------------------------
def recommend_movies(user_input):
    user_input = user_input.lower()
    
    # Transform user input to vector
    user_vec = vectorizer.transform([user_input])
    
    # Calculate similarity
    similarity = cosine_similarity(user_vec, tfidf_matrix)
    
    # Get top 5 similar movies
    indices = similarity[0].argsort()[-5:][::-1]
    
    recommended = movies.iloc[indices]['Title'].tolist()
    
    if len(recommended) == 0:
        return ["Sorry, I couldn’t find any similar movies."]
    return recommended

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🎬 Smart Movie Recommendation Chatbot")
st.write("Hi! I’m your smart movie bot 🤖 — type a *movie name* or *genre* to get recommendations!")

user_input = st.text_input("You:", placeholder="e.g. Action, Inception, Comedy...")

if user_input:
    st.subheader("🎥 Recommended Movies:")
    recos = recommend_movies(user_input)
    for r in recos:
        st.write(f"- {r}")

st.markdown("---")
st.caption("Made with ❤️ by Rishav | Internship Project")

#run - streamlit run movie_chatbot_app.py


