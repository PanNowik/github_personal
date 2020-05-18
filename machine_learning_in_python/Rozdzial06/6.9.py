# Wczytanie bibliotek.
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Utworzenie tekstu.
text_data = np.array(['I love Brazil. Brazil!',
                      'Sweden is best',
                      'Germany beats both'])

# Utworzenie macierzy cech do użycia z tf-idf.
tfidf = TfidfVectorizer()
feature_matrix = tfidf.fit_transform(text_data)

# Wyświetlenie macierzy cech tf-idf.
feature_matrix
