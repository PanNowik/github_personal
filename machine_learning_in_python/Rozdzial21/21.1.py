# Wczytanie bibliotek.
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.externals import joblib

# Wczytanie danych.
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Utworzenie obiektu klasyfikatora drzewa decyzyjnego.
classifer = RandomForestClassifier()

# Wytrenowanie modelu.
model = classifer.fit(features, target)

# Zapisanie modelu w pliku typu pickle.
joblib.dump(model, "model.pkl")
