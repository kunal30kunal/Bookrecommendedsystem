import pandas as pd

def load_data():
    books = pd.read_csv("data/BX_Books.csv", sep=";", encoding="latin-1", on_bad_lines="skip", low_memory=False)
    users = pd.read_csv("data/BX-Users.csv", sep=";", encoding="latin-1", on_bad_lines="skip", low_memory=False)
    ratings = pd.read_csv("data/BX-Book-Ratings.csv", sep=";", encoding="latin-1", on_bad_lines="skip", low_memory=False)
    return books, users, ratings


def filter_data(books, users, ratings):
    # Filter out implicit ratings (0)
    ratings = ratings[ratings['Book-Rating'] > 0]

    # Filter users who rated more than X books
    active_users = ratings['User-ID'].value_counts()
    active_users = active_users[active_users > 10].index
    ratings = ratings[ratings['User-ID'].isin(active_users)]

    # Filter books with more than X ratings
    popular_books = ratings['ISBN'].value_counts()
    popular_books = popular_books[popular_books > 10].index
    ratings = ratings[ratings['ISBN'].isin(popular_books)]

    # Update books and users accordingly
    books = books[books['ISBN'].isin(ratings['ISBN'])]
    users = users[users['User-ID'].isin(ratings['User-ID'])]

    return books, users, ratings


def get_pivot_table(ratings):
    pivot_table = ratings.pivot_table(index='ISBN', columns='User-ID', values='Book-Rating')
    pivot_table.fillna(0, inplace=True)
    return pivot_table



