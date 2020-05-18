# Wczytanie bibliotek.
from sklearn.decomposition import NMF
from sklearn import datasets

# Wczytanie danych.
digits = datasets.load_digits()

# Wczytanie macierzy cech.
features = digits.data

# Zdefiniowanie, dopasowanie i zastosowanie NMF.
nmf = NMF(n_components=10, random_state=1)
features_nmf = nmf.fit_transform(features)

# Wyświetlenie wyników.
print("Początkowa liczba cech:", features.shape[1])
print("Liczba cech po redukcji:", features_nmf.shape[1])
