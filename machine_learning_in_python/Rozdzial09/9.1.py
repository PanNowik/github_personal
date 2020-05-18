# Wczytanie bibliotek.
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import datasets

# Wczytanie danych.
digits = datasets.load_digits()

# Standaryzacja macierzy cech.
features = StandardScaler().fit_transform(digits.data)

# Zdefiniowanie analizy głównych składowych, zapewniającej 99-procentową wariancję.
pca = PCA(n_components=0.99, whiten=True)

# Przeprowadzenie analizy głównych składowych.
features_pca = pca.fit_transform(features)

# Wyświetlenie wyników.
print("Początkowa liczba cech:", features.shape[1])
print("Liczba cech po redukcji:", features_pca.shape[1])
