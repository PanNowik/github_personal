# Wczytanie bibliotek.
from sklearn.datasets import load_boston
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split

# Wczytanie danych.
boston = load_boston()

# Utworzenie cech.
features, target = boston.data, boston.target

# Podział na zbiory uczący i testowy.
features_train, features_test, target_train, target_test = train_test_split(
    features, target, random_state=0)

# Utworzenie sztucznego regresora.
dummy = DummyRegressor(strategy='mean')

# "Wytrenowanie" sztucznego regresora.
dummy.fit(features_train, target_train)

# Pobranie kwadratu wartości.
dummy.score(features_test, target_test)
