# Wczytanie bibliotek.
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.feature_selection import SelectFromModel

# Wczytanie danych.
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Utworzenie klasyfikatora losowego lasu.
randomforest = RandomForestClassifier(random_state=0, n_jobs=-1)

# Utworzenie obiektu wybierającego cechy o wadze większej
# niż lub równej podanej wartości progowej.
selector = SelectFromModel(randomforest, threshold=0.3)

# Przygotowanie nowej macierzy cech.
features_important = selector.fit_transform(features, target)

# Wytrenowanie losowego lasu za pomocą najważniejszych cech.
model = randomforest.fit(features_important, target)
