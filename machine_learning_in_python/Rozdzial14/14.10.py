# Wczytanie bibliotek.
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets

# Wczytanie danych.
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Utworzenie obiektu klasyfikatora drzewa AdaBoostClassifier.
adaboost = AdaBoostClassifier(random_state=0)

# Wytrenowanie modelu.
model = adaboost.fit(features, target)
