# Wczytanie bibliotek.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets

# Wczytanie danych.
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Utworzenie obiektu klasyfikatora losowego lasu.
randomforest = RandomForestClassifier(random_state=0, n_jobs=-1)

# Wytrenowanie modelu.
model = randomforest.fit(features, target)

# Ustalenie wagi cechy.
importances = model.feature_importances_

# Posortowanie wagi cech w kolejności malejącej.
indices = np.argsort(importances)[::-1]

# Ponowne rozmieszczenie nazw cech w taki sposób, aby zostały dopasowane do posortowanych wag cech.
names = [iris.feature_names[i] for i in indices]

# Utworzenie wykresu.
plt.figure()

# Utworzenie tytułu wykresu.
plt.title("Waga cech")

# Dodanie słupków.
plt.bar(range(features.shape[1]), importances[indices])

# Dodanie nazw cech jako etykiet na osi X.
plt.xticks(range(features.shape[1]), names, rotation=90)

# Wyświetlenie wykresu.
plt.show()
