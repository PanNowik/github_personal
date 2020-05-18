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
penalty = ["l1", "l2"]

# Zdefiniowanie zakresu potencjalnych wartości dla C.
C = np.logspace(0, 4, 1000)

# Zdefiniowanie opcji hiperparametru.
hyperparameters = dict(C=C, penalty=penalty)

# Zdefiniowanie siatki przeszukiwania.
gridsearch = GridSearchCV(logistic, hyperparameters, cv=5, n_jobs=-1, verbose=1)

# Dopasowanie siatki przeszukiwania.
best_model = gridsearch.fit(features, target)
