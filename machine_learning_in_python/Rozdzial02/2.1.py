# Wczytanie zbiorów danych biblioteki scikit-learn.
from sklearn import datasets

# Wczytanie zbioru danych w postaci cyfr.
digits = datasets.load_digits()

# Utworzenie macierzy cech.
features = digits.data

# Utworzenie wektora docelowego.
target = digits.target

# Wyświetlenie pierwszej obserwacji.
features[0]
