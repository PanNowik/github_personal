# Wczytanie bibliotek.
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# Wczytanie danych.
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Utworzenie wysoce niezrównoważonej klasy przez usunięcie pierwszych 40 obserwacji.
features = features[40:,:]
target = target[40:]

# Utworzenie wektora docelowego wskazującego, czy element jest klasy 0, czy 1.
target = np.where((target == 0), 0, 1)

# Standaryzowanie cech.
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# Utworzenie obiektu klasyfikatora drzewa decyzyjnego.
logistic_regression = LogisticRegression(random_state=0, class_weight="balanced")

# Wytrenowanie modelu.
model = logistic_regression.fit(features_standardized, target)
