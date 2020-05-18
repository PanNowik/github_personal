# Wczytanie bibliotek.
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import datasets

# Wczytanie danych.
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Utworzenie egzemplarza typu StandardScaler.
standardizer = StandardScaler()

# Standaryzacja cech.
features_standardized = standardizer.fit_transform(features)

# Wytrenowanie klasyfikatora sąsiedztwa na podstawie promienia.
rnn = RadiusNeighborsClassifier(
    radius=.5, n_jobs=-1).fit(features_standardized, target)

# Utworzenie dwóch obserwacji.
new_observations = [[ 1,  1,  1,  1]]

# Prognozowanie klasy tych dwóch obserwacji.
rnn.predict(new_observations)
