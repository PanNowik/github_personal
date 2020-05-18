# Wczytanie bibliotek.
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np

#Wczytanie danych zawierających tylko dwie klasy.
iris = datasets.load_iris()
features = iris.data[:100,:]
target = iris.target[:100]

# Standaryzowanie cech.
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# Utworzenie obiektu klasyfikatora wektora nośnego.
svc = SVC(kernel="linear", random_state=0)

# Wytrenowanie klasyfikatora.
model = svc.fit(features_standardized, target)

# Wyświetlenie wektorów nośnych.
model.support_vectors_
