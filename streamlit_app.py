import streamlit as st
from recommender.data_preprocessing import load_data, filter_data, get_pivot_table
from recommender.recommender_engine import train_knn_model, recommend_books
import pandas as pd
import os

st.set_page_config(page_title="Book Recommender", layout="wide")

# Load and process data
@st.cache_data
def setup():
    books, users, ratings = load_data()
    books, users, ratings = filter_data(books, users, ratings)
    pivot_table = get_pivot_table(ratings)
    model = train_knn_model(pivot_table)
    return books, pivot_table, model

books, pivot_table, model = setup()

# UI Title
st.title("📚 Book Recommender System")

# Select Book
book_isbns = pivot_table.index.tolist()
book_titles = books[books['ISBN'].isin(book_isbns)][['ISBN', 'Book-Title']].drop_duplicates()

# Dropdown
selected_title = st.selectbox("Choose a book you like:", book_titles['Book-Title'].values)

# Get ISBN for selected title
selected_isbn = book_titles[book_titles['Book-Title'] == selected_title]['ISBN'].values[0]

# Slider to choose number of recommendations
num_recs = st.slider("Number of recommendations", 3, 10, 5)

# Recommend books
if st.button("Get Recommendations"):
    recommended_isbns = recommend_books(selected_isbn, pivot_table, model, n_recommendations=num_recs)

    st.subheader("📖 Recommended Books:")
    cols = st.columns(5)

    for idx, isbn in enumerate(recommended_isbns):
        book = books[books['ISBN'] == isbn].drop_duplicates()
        if book.empty:
            continue
        title = book['Book-Title'].values[0]
        author = book['Book-Author'].values[0]
        image_url = book['Image-URL-M'].values[0]  # medium size image

        with cols[idx % 5]:
            st.image(image_url, width=120)
            st.write(f"**{title}**")
            st.caption(f"by {author}")
