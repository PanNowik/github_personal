# Wczytanie bibliotek.
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd

# Wczytanie danych.
iris = datasets.load_iris()

# Utworzenie macierzy cech.
features = iris.data

# Utworzenie wektora docelowego.
target = iris.target

# Utworzenie listy klas docelowych.
class_names = iris.target_names

# Utworzenie zbiorów uczącego i testowego.
features_train, features_test, target_train, target_test = train_test_split(
    features, target, random_state=1)

# Utworzenie regresji logistycznej.
classifier = LogisticRegression()

# Wytrenowanie modelu i wygenerowanie prognoz.
target_predicted = classifier.fit(features_train,
    target_train).predict(features_test)

# Utworzenie macierzy pomyłek.
matrix = confusion_matrix(target_test, target_predicted)

# Utworzenie obiektu DataFrame biblioteki pandas.
dataframe = pd.DataFrame(matrix, index=class_names, columns=class_names)

# Utworzenie mapy cieplnej.
sns.heatmap(dataframe, annot=True, cbar=None, cmap="Blues")
plt.title("Macierz pomyłek"), plt.tight_layout()
plt.ylabel("Klasy prawdziwe"), plt.xlabel("Klasy prognozowane")
plt.show()
