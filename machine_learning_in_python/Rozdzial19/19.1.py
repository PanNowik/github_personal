# Wczytanie bibliotek.
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Wczytanie danych.
iris = datasets.load_iris()
features = iris.data

# Standaryzowanie cech.
scaler = StandardScaler()
features_std = scaler.fit_transform(features)

# Utworzenie obiektu k średnich.
cluster = KMeans(n_clusters=3, random_state=0, n_jobs=-1)

# Wytrenowanie modelu.
model = cluster.fit(features_std)
