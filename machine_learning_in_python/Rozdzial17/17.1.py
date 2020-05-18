# Wczytanie bibliotek.
from sklearn.svm import LinearSVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np

# Wczytanie danych zawierających tylko dwie klasy i dwie cechy.
iris = datasets.load_iris()
features = iris.data[:100,:2]
target = iris.target[:100]

# Standaryzowanie cech.
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# Utworzenie klasyfikatora wektora nośnego.
svc = LinearSVC(C=1.0)

# Wytrenowanie modelu.
model = svc.fit(features_standardized, target)
