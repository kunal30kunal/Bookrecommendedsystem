from sklearn.neighbors import NearestNeighbors


# Train the model
def train_knn_model(pivot_table):
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(pivot_table)
    return model


# Recommend similar books by ISBN
def recommend_books(book_isbn, pivot_table, model, n_recommendations=5):
    if book_isbn not in pivot_table.index:
        return []

    distances, indices = model.kneighbors(
        pivot_table.loc[[book_isbn]], n_neighbors=n_recommendations + 1
    )

    similar_books = pivot_table.index[indices.flatten()[1:]]  # skip the book itself
    return list(similar_books)
