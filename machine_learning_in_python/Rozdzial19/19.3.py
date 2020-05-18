# Wczytanie bibliotek.
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MeanShift

# Wczytanie danych.
iris = datasets.load_iris()
features = iris.data

# Standaryzowanie cech.
scaler = StandardScaler()
features_std = scaler.fit_transform(features)

# Utworzenie obiektu typu MeanShift.
cluster = MeanShift(n_jobs=-1)

# Wytrenowanie modelu.
model = cluster.fit(features_std)
