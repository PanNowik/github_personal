# Wczytanie bibliotek.
import pydotplus
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from IPython.display import Image
from sklearn import tree

# Wczytanie danych.
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Utworzenie obiektu klasyfikatora drzewa decyzyjnego.
decisiontree = DecisionTreeClassifier(random_state=0)

# Wytrenowanie modelu.
model = decisiontree.fit(features, target)

# Utworzenie danych w formacie DOT.
dot_data = tree.export_graphviz(decisiontree,
                                out_file=None,
                                feature_names=iris.feature_names,
                                class_names=iris.target_names)

# Wygenerowanie drzewa w postaci wizualnej.
graph = pydotplus.graph_from_dot_data(dot_data)

# Wyświetlenie wygenerowanego drzewa.
Image(graph.create_png())
