# Wczytanie bibliotek.
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets

# Wczytanie danych.
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Utworzenie obiektu klasyfikatora losowego lasu.
randomforest = RandomForestClassifier(random_state=0, n_jobs=-1)

# Wytrenowanie modelu.
model = randomforest.fit(features, target)
