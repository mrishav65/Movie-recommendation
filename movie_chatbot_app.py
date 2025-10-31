import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@st.cache_data
def load_data():
    df = pd.read_csv("movies.csv")
    df.columns = df.columns.str.strip()
    df['Series_Title'] = df['Series_Title'].astype(str)
    df['Overview'] = df['Overview'].astype(str)
    df['Genre'] = df['Genre'].astype(str)
    df['Director'] = df['Director'].astype(str)
    return df

movies = load_data()


movies['combined_features'] = (
    movies['Genre'] + " " +
    movies['Director'] + " " +
    movies['Overview']
)


vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(movies['combined_features'])


similarity = cosine_similarity(tfidf_matrix)


st.title("🎥 Movie Recommendation System")
st.markdown("Get movie recommendations based on your favorite movie!")


movie_list = movies['Series_Title'].sort_values().unique()
selected_movie = st.selectbox("Choose a movie you like:", movie_list)


def recommend(movie_name, n=5):
    idx = movies[movies['Series_Title'] == movie_name].index[0]
    distances = similarity[idx]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:n+1]

    recommendations = []
    for i in movie_list:
        recommendations.append({
            "Title": movies.iloc[i[0]].Series_Title,
            "Genre": movies.iloc[i[0]].Genre,
            "Rating": movies.iloc[i[0]].IMDB_Rating,
            "Poster": movies.iloc[i[0]].Poster_Link,
            "Overview": movies.iloc[i[0]].Overview
        })
    return recommendations


if st.button("Show Recommendations"):
    st.subheader(f"🎬 Movies similar to: {selected_movie}")
    recs = recommend(selected_movie)
    for rec in recs:
        st.markdown(f"### {rec['Title']} ({rec['Rating']}/10 ⭐)")
        st.image(rec['Poster'], width=200)
        st.write(f"**Genre:** {rec['Genre']}")
        st.write(f"**Overview:** {rec['Overview']}")
        st.write("---")

st.caption("Built with ❤️ by Rishav using Streamlit + Machine Learning")

