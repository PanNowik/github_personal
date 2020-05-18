# Wczytanie bibliotek.
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif

# Wczytanie danych.
iris = load_iris()
features = iris.data
target = iris.target

# Konwersja na dane kategoryzujące przez konwersję danych na liczby całkowite.
features = features.astype(int)

# Wybór dwóch cech o największych wartościach danych statystycznych chi-kwadrat.
chi2_selector = SelectKBest(chi2, k=2)
features_kbest = chi2_selector.fit_transform(features, target)

# Wyświetlenie wyników.
print("Początkowa liczba cech:", features.shape[1])
print("Liczba cech po redukcji:", features_kbest.shape[1])
