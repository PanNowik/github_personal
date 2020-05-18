# Wczytanie bibliotek.
from sklearn import datasets
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Wczytanie danych.
iris = datasets.load_iris()
features = iris.data

# Utworzenie egzemplarza StandardScaler.
standardizer = StandardScaler()

# Standaryzacja cech.
features_standardized = standardizer.fit_transform(features)

# Dwóch najbliższych sąsiadów.
nearest_neighbors = NearestNeighbors(n_neighbors=2).fit(features_standardized)

# Utworzenie obserwacji.
new_observation = [ 1,  1,  1,  1]

# Obliczenie odległości i odszukanie indeksów najbliższych sąsiadów obserwacji.
distances, indices = nearest_neighbors.kneighbors([new_observation])

# Wyświetlenie najbliższych sąsiadów.
features_standardized[indices]
