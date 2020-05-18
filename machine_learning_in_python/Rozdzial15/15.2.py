# Wczytanie bibliotek.
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import datasets

# Wczytanie danych.
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Utworzenie egzemplarza StandardScaler.
standardizer = StandardScaler()

# Standaryzacja cech.
X_std = standardizer.fit_transform(X)

# Wytrenowanie klasyfikatora KNN wraz z pięcioma najbliższymi sąsiadami.
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1).fit(X_std, y)

# Utworzenie dwóch obserwacji.
new_observations = [[ 0.75,  0.75,  0.75,  0.75],
                    [ 1,  1,  1,  1]]

# Prognozowanie klasy tych dwóch obserwacji.
knn.predict(new_observations)
