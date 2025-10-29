import streamlit as st
import pandas as pd
import spacy
import random

# Load the NLP model
nlp = spacy.load("en_core_web_sm")

# Load your movie dataset
movies = pd.read_csv("movies.csv")

# Clean missing data
for col in ['Genre', 'Director', 'Star1', 'Star2', 'Star3', 'Star4', 'Series_Title', 'Released_Year', 'IMDB_Rating']:
    if col in movies.columns:
        movies[col] = movies[col].fillna('').astype(str)

# ----- Recommendation Functions -----

def recommend_by_genre(genre):
    results = movies[movies['Genre'].str.contains(genre, case=False, na=False)]
    if results.empty:
        return []
    return results.sort_values(by='IMDB_Rating', ascending=False).head(5)

def recommend_by_director(name):
    results = movies[movies['Director'].str.contains(name, case=False, na=False)]
    if results.empty:
        return []
    return results.sort_values(by='IMDB_Rating', ascending=False).head(5)

def recommend_by_actor(name):
    results = movies[
        movies['Star1'].str.contains(name, case=False, na=False) |
        movies['Star2'].str.contains(name, case=False, na=False) |
        movies['Star3'].str.contains(name, case=False, na=False) |
        movies['Star4'].str.contains(name, case=False, na=False)
    ]
    if results.empty:
        return []
    return results.sort_values(by='IMDB_Rating', ascending=False).head(5)

def recommend_by_year(year, after=True):
    try:
        year = int(year)
    except ValueError:
        return []
    if after:
        results = movies[movies['Released_Year'].astype(int) >= year]
    else:
        results = movies[movies['Released_Year'].astype(int) <= year]
    if results.empty:
        return []
    return results.sort_values(by='IMDB_Rating', ascending=False).head(5)

def recommend_by_rating(min_rating):
    try:
        min_rating = float(min_rating)
    except ValueError:
        return []
    results = movies[movies['IMDB_Rating'].astype(float) >= min_rating]
    if results.empty:
        return []
    return results.sort_values(by='IMDB_Rating', ascending=False).head(5)

def random_movie():
    return movies.sample(1)

# ----- NLP Intent Detection -----

def detect_intent(user_input):
    doc = nlp(user_input.lower())
    genres = ["action", "drama", "comedy", "thriller", "romance", "horror", "sci-fi", "animation", "adventure"]

    for token in doc:
        if token.text in genres:
            return "genre", token.text

    if "director" in user_input or "by" in user_input:
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return "director", ent.text

    if "actor" in user_input or "starring" in user_input or "with" in user_input:
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return "actor", ent.text

    if "after" in user_input:
        for token in doc:
            if token.like_num:
                return "year_after", token.text

    if "before" in user_input:
        for token in doc:
            if token.like_num:
                return "year_before", token.text

    if "rating" in user_input or "above" in user_input:
        for token in doc:
            if token.like_num:
                return "rating", token.text

    if "random" in user_input or "suggest" in user_input:
        return "random", None

    if "top" in user_input or "best" in user_input:
        return "top", None

    return "unknown", None

# ----- Streamlit UI -----

st.set_page_config(page_title="🎬 Movie Recommendation Chatbot", page_icon="🎥", layout="centered")
st.title("🎬 Movie Recommendation Chatbot")
st.write("Ask me anything like:")
st.markdown("""
- *Recommend me action movies*  
- *Movies by Christopher Nolan*  
- *Actor Leonardo DiCaprio*  
- *Films after 2010*  
- *Rating above 8*  
- *Give me a random movie*  
""")

user_input = st.text_input("You:", placeholder="Type your message here...")

if user_input:
    intent, entity = detect_intent(user_input)
    st.markdown("**🤖 Chatbot:**")

    if intent == "genre":
        df = recommend_by_genre(entity)
    elif intent == "director":
        df = recommend_by_director(entity)
    elif intent == "actor":
        df = recommend_by_actor(entity)
    elif intent == "year_after":
        df = recommend_by_year(entity, after=True)
    elif intent == "year_before":
        df = recommend_by_year(entity, after=False)
    elif intent == "rating":
        df = recommend_by_rating(entity)
    elif intent == "random":
        df = random_movie()
    elif intent == "top":
        df = movies.sort_values(by='IMDB_Rating', ascending=False).head(5)
    else:
        st.warning("I didn’t understand that. Try asking about genre, director, actor, year, or rating.")
        df = None

    if df is not None and not df.empty:
        for _, row in df.iterrows():
            st.markdown(f"🎥 **{row.Series_Title}** ({row.Released_Year}) — ⭐ {row.IMDB_Rating} <br>🎭 *{row.Genre}* <br>🎬 Directed by {row.Director}", unsafe_allow_html=True)
            st.write("---")


#run - streamlit run movie_chatbot_app.py
