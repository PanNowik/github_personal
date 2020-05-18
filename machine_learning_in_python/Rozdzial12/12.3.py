# Wczytanie bibliotek.
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# Zdefiniowanie wartości zalążka.
np.random.seed(0)

# Wczytanie danych.
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Utworzenia egzemplarza typu Pipeline.
pipe = Pipeline([("classifier", RandomForestClassifier())])

# Utworzenie słownika wybranych algorytmów uczenia maszynowego i ich hiperparametrów.
search_space = [{"classifier": [LogisticRegression()],
                 "classifier__penalty": ['l1', 'l2'],
                 "classifier__C": np.logspace(0, 4, 10)},
                {"classifier": [RandomForestClassifier()],
                 "classifier__n_estimators": [10, 100, 1000],
                 "classifier__max_features": [1, 2, 3]}]

# Zdefiniowanie siatki przeszukiwania.
gridsearch = GridSearchCV(pipe, search_space, cv=5, verbose=0)

# Dopasowanie siatki przeszukiwania.
best_model = gridsearch.fit(features, target)
