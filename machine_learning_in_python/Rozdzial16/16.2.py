# Wczytanie bibliotek.
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# Wczytanie danych.
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Standaryzowanie cech.
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# Utworzenie obiektu regresji logistycznej wykorzystującej technikę typu jeden przeciwko reszcie.
logistic_regression = LogisticRegression(random_state=0, multi_class="ovr")

# Wytrenowanie modelu.
model = logistic_regression.fit(features_standardized, target)
