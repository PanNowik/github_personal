# Wczytanie bibliotek.
from sklearn import datasets
from sklearn.feature_selection import VarianceThreshold

# Wczytanie pewnego zbioru danych.
iris = datasets.load_iris()

# Utworzenie cech i zdefiniowanie celu.
features = iris.data
target = iris.target

# Zdefiniowanie wartości progowej.
thresholder = VarianceThreshold(threshold=.5)

# Utworzenie macierzy cech o wysokiej wariancji.
features_high_variance = thresholder.fit_transform(features)

# Wyświetlenie macierzy cech o wysokiej wariancji.
features_high_variance[0:3]
