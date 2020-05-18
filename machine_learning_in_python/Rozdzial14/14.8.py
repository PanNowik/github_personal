# Wczytanie bibliotek.
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets

# Wczytanie danych.
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Zdefiniowanie klasy jako wysoce niezrównoważonej przez usunięcie pierwszych 40 obserwacji.
features = features[40:,:]
target = target[40:]

# Utworzenie wektora docelowego wskazującego, czy klasa obserwacji to 0. W przeciwnym przypadku będzie to klasa 1.
target = np.where((target == 0), 0, 1)

# Utworzenie obiektu klasyfikatora losowego lasu.
randomforest = RandomForestClassifier(
    random_state=0, n_jobs=-1, class_weight="balanced")

# Wytrenowanie modelu.
model = randomforest.fit(features, target)
