# Wczytanie bibliotek.
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# Wczytanie danych.
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Standaryzowanie cech.
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# Utworzenie obiektu regresji logistycznej.
logistic_regression = LogisticRegression(random_state=0, solver="sag")

# Wytrenowanie modelu.
model = logistic_regression.fit(features_standardized, target)
