import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the datasets
books = pd.read_parquet('Books2.parquet')
ratings = pd.read_csv('Ratings.csv')
users = pd.read_csv('Users.csv')

# Data preprocessing
users = users[(users.Age < 100) | (users.Age.isna()) | (users.Age > 2)]
users.Age.fillna(users.Age.mean(), inplace=True)
books.dropna(inplace=True)
# Clean and format the book titles
books['Book-Title'] = books['Book-Title'].str.title()  
books['Book-Title'] = books['Book-Title'].str.replace(r'\s*\(Paperback\)', '', regex=True).str.strip()  # Remove '(Paperback)'
# books['Book-Title'] = books['Book-Title'].str.replace(r'\b\b', '', regex=True).str.strip()  # Remove 'Paperback'

ratings_books = ratings.merge(books, on='ISBN')

df = ratings_books.groupby('Book-Title').agg(
    Book_Rating_Count=('Book-Rating', 'count'),
    Book_Rating_Mean=('Book-Rating', lambda x: round(x.mean(), 2))
).reset_index()

readers = ratings_books.groupby('User-ID').filter(lambda x: len(x) > 100)
fin_df = readers.groupby('Book-Title').filter(lambda x: len(x) > 25)

# Create a pivot table
df = fin_df.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
df.fillna(0, inplace=True)

# Calculate cosine similarity
similarity_scores = cosine_similarity(df)

# Recommendation function
def recommend(book_name):
    index = np.where(df.index == book_name)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:6]
    
    recommendations = []
    for i in similar_items:
        temp_df = books[books['Book-Title'] == df.index[i[0]]]
        title = temp_df.drop_duplicates('Book-Title')['Book-Title'].values[0]
        author = temp_df.drop_duplicates('Book-Title')['Book-Author'].values[0]
        image_url = temp_df.drop_duplicates('Book-Title')['Image-URL-L'].values[0]
        recommendations.append({"Title": title, "Author": author, "Image-URL": image_url})
    
    return recommendations

# Streamlit UI
st.title("Next Read")

# Dropdown for book selection
book_options = df.index.tolist()
selected_book = st.selectbox("Find Your Next Read based on your preferences:", book_options)

if st.button("Recommend"):
    recommendations = recommend(selected_book)
    
    # Display the recommendations
    st.subheader("Recommended Books:")
    for rec in recommendations:
        st.image(rec['Image-URL'], caption=f"{rec['Title']} by {rec['Author']}", width=180)
        st.write("---")