# Wczytanie bibliotek.
from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

# Wczytanie danych.
iris = load_iris()

# Utworzenie wektora docelowego i macierzy cech.
features, target = iris.data, iris.target

# Podział na zbiory uczący i testowy.
features_train, features_test, target_train, target_test = train_test_split(
features, target, random_state=0)

# Utworzenie sztucznego klasyfikatora.
dummy = DummyClassifier(strategy='uniform', random_state=1)

# "Wytrenowanie" modelu.
dummy.fit(features_train, target_train)

# Pobranie wyniku dokładności.
dummy.score(features_test, target_test)
