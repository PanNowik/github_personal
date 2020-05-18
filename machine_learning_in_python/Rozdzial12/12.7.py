# Wczytanie bibliotek.
import numpy as np
from sklearn import linear_model, datasets
from sklearn.model_selection import GridSearchCV, cross_val_score

# Wczytanie danych.
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Zdefiniowanie regresji logistycznej.
logistic = linear_model.LogisticRegression()

# Zdefiniowanie zakresu 20 potencjalnych wartości dla C.
C = np.logspace(0, 4, 20)

# Zdefiniowanie opcji hiperparametru.
hyperparameters = dict(C=C)

# Zdefiniowanie siatki przeszukiwania.
gridsearch = GridSearchCV(logistic, hyperparameters, cv=5, n_jobs=-1, verbose=0)

# Przeprowadzenie zagnieżdżonego sprawdzianu krzyżowego i wyświetlenie wartości średniej.
cross_val_score(gridsearch, features, target).mean()
