# Wczytanie biblioteki.
from sklearn.feature_extraction import DictVectorizer

# Utworzenie słownika.
data_dict = [{"czerwony": 2, "niebieski": 4},
             {"czerwony": 4, "niebieski": 3},
             {"czerwony": 1, "żółty": 2},
             {"czerwony": 2, "żółty": 2}]

# Utworzenie słownika, czyli egzemplarza typu DictVectorizer.
dictvectorizer = DictVectorizer(sparse=False)

# Konwersja słownika na macierz cech.
features = dictvectorizer.fit_transform(data_dict)

# Wyświetlenie macierzy cech.
features
