# Wczytanie bibliotek.
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

# Wczytanie danych.
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Utworzenie obiektu naiwnego klasyfikatora bayesowskiego Gaussa.
classifer = GaussianNB()

# Wytrenowanie modelu.
model = classifer.fit(features, target)
