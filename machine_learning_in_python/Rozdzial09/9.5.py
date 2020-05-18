# Wczytanie bibliotek.
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from sklearn import datasets
import numpy as np

# Wczytanie danych.
digits = datasets.load_digits()

# Standaryzacja macierzy cech.
features = StandardScaler().fit_transform(digits.data)

# Utworzenie macierzy rzadkiej.
features_sparse = csr_matrix(features)

# Zdefiniowanie operacji TSVD.
tsvd = TruncatedSVD(n_components=10)

# Przeprowadzenie operacji TSVD na macierzy rzadkiej.
features_sparse_tsvd = tsvd.fit(features_sparse).transform(features_sparse)

# Wyświetlenie wyników.
print("Początkowa liczba cech:", features_sparse.shape[1])
print('"Liczba cech po redukcji:", features_sparse_tsvd.shape[1])
