# Wczytanie bibliotek.
from sklearn.linear_model import Ridge
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

# Wczytanie danych.
boston = load_boston()
features = boston.data
target = boston.target

# Standaryzacja cech.
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# Zdefiniowanie regresji grzbietowej z wartością alfa.
regression = Ridge(alpha=0.5)

# Wyznaczenie regresji liniowej.
model = regression.fit(features_standardized, target)
