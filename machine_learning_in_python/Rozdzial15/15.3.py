# Wczytanie bibliotek.
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV

# Wczytanie danych.
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Utworzenie egzemplarza typu StandardScaler.
standardizer = StandardScaler()

# Standaryzacja cech.
features_standardized = standardizer.fit_transform(features)

# Utworzenie klasyfikatora KNN.
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)

# Utworzenie egzemplarza typu Pipeline.
pipe = Pipeline([("standardizer", standardizer), ("knn", knn)])

# Przygotowanie przestrzeni dla prognozowanych wartości.
search_space = [{"knn__n_neighbors": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}]

# Zdefiniowanie metody przeszukiwania siatki.
classifier = GridSearchCV(
    pipe, search_space, cv=5, verbose=0).fit(features_standardized, target)
