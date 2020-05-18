# Wczytanie bibliotek.
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

# Wczytanie danych.
iris = datasets.load_iris()
features = iris.data

# Standaryzowanie cech.
scaler = StandardScaler()
features_std = scaler.fit_transform(features)

# Utworzenie obiektu typu MeanShift.
cluster = AgglomerativeClustering(n_clusters=3)

# Wytrenowanie modelu.
model = cluster.fit(features_std)
