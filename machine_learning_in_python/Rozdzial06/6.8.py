# Wczytanie biblioteki.
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Utworzenie tekstu.
text_data = np.array(['I love Brazil. Brazil!',
                      'Sweden is best',
                      'Germany beats both'])

# Utworzenie macierzy cech zbioru słów.
count = CountVectorizer()
bag_of_words = count.fit_transform(text_data)

# Wyświetlenie macierzy cech.
bag_of_words
