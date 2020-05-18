# Wczytanie bibliotek.
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np

# Wczytanie danych.
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Standaryzowanie cech.
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# Utworzenie obiektu klasyfikatora wektora nośnego.
svc = SVC(kernel="linear", probability=True, random_state=0)

# Wytrenowanie klasyfikatora.
model = svc.fit(features_standardized, target)

# Utworzenie nowej obserwacji.
new_observation = [[.4, .4, .4, .4]]

# Wyświetlenie prawdopodobieństwa prognozy.
model.predict_proba(new_observation)
