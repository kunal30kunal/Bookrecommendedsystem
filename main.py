import random
from recommender.data_preprocessing import load_data, filter_data, get_pivot_table
from recommender.recommender_engine import train_knn_model, recommend_books

# Step 1: Load and clean the data
books, users, ratings = load_data()
books, users, ratings = filter_data(books, users, ratings)

# Step 2: Create pivot table
pivot_table = get_pivot_table(ratings)
print("Pivot shape:", pivot_table.shape)

# Step 3: Train KNN model
model = train_knn_model(pivot_table)

# Step 4: Pick a random valid ISBN from pivot table
sample_isbn = random.choice(pivot_table.index.tolist())
print(f"\nUsing random valid ISBN for recommendation: {sample_isbn}")

# Step 5: Get recommendations
recommended_isbns = recommend_books(sample_isbn, pivot_table, model)

# Step 6: Show book titles
print("\nRecommendations:")
for isbn in recommended_isbns:
    title = books[books['ISBN'] == isbn]['Book-Title'].values[0]
    print(f"{isbn}: {title}")
