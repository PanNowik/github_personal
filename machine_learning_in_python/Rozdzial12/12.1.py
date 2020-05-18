# Wczytanie bibliotek.
import numpy as np
from sklearn import linear_model, datasets
from sklearn.model_selection import GridSearchCV

# Wczytanie danych.
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Zdefiniowanie regresji logistycznej.
logistic = linear_model.LogisticRegression()

# Zdefiniowanie zakresu wartości hiperparametru penalty.
penalty = ['l1', 'l2']

# Zdefiniowanie zakresu wartości hiperparametru regularyzacji.
C = np.logspace(0, 4, 10)

# Utworzenie słownika wartości hiperparametrów.
hyperparameters = dict(C=C, penalty=penalty)

# Zdefiniowanie siatki przeszukiwania.
gridsearch = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0)

# Dopasowanie siatki przeszukiwania.
best_model = gridsearch.fit(features, target)
