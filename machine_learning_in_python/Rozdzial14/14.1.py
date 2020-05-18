# Wczytanie bibliotek.
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets

# Wczytanie danych.
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Utworzenie obiektu klasyfikatora drzewa decyzyjnego.
decisiontree = DecisionTreeClassifier(random_state=0)

# Wytrenowanie modelu.
model = decisiontree.fit(features, target)
