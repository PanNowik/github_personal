# Wczytanie bibliotek.
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

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
model = classifier.fit(features_train, target_train)
target_predicted = model.predict(features_test)

# Wygenerowanie raportu klasyfikatora.
print(classification_report(target_test,
                            target_predicted,
                            target_names=class_names))
