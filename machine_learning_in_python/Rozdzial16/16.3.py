# Wczytanie bibliotek.
from sklearn.linear_model import LogisticRegressionCV
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# Wczytanie danych.
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Standaryzowanie cech.
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# Utworzenie obiektu klasyfikatora drzewa decyzyjnego.
logistic_regression = LogisticRegressionCV(
    penalty='l2', Cs=10, random_state=0, n_jobs=-1)

# Wytrenowanie modelu.
model = logistic_regression.fit(features_standardized, target)
