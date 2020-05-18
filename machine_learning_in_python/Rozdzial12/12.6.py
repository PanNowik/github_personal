# Wczytanie bibliotek.
from sklearn import linear_model, datasets

# Wczytanie danych.
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Zdefiniowanie sprawdzianu krzyżowego regresji logistycznej.
logit = linear_model.LogisticRegressionCV(Cs=100)

# Wytrenowanie modelu.
logit.fit(features, target)
