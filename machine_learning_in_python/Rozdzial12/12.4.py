# Wczytanie bibliotek.
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Zdefiniowanie wartości zalążka.
np.random.seed(0)

# Wczytanie danych.
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Utworzenie obiektu przetwarzania danych obejmującego egzemplarze typów StandardScaler i PCA.
preprocess = FeatureUnion([("std", StandardScaler()), ("pca", PCA())])

# Utworzenia egzemplarza typu Pipeline.
pipe = Pipeline([("preprocess", preprocess),
                 ("classifier", LogisticRegression())])

# Przygotowanie miejsca dla wartości.
search_space = [{"preprocess__pca__n_components": [1, 2, 3],
                 "classifier__penalty": ["l1", "l2"],
                 "classifier__C": np.logspace(0, 4, 10)}]

# Zdefiniowanie siatki przeszukiwania.
clf = GridSearchCV(pipe, search_space, cv=5, verbose=0, n_jobs=-1)

# Dopasowanie siatki przeszukiwania.
best_model = clf.fit(features, target)
