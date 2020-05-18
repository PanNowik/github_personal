# Wczytanie bibliotek.
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np

#Wczytanie danych zawierających tylko dwie klasy.
iris = datasets.load_iris()
features = iris.data[:100,:]
target = iris.target[:100]

# Zdefiniowanie klasy jako wysoce niezrównoważonej przez usunięcie pierwszych 40 obserwacji.
features = features[40:,:]
target = target[40:]

# Utworzenie wektora docelowego wskazującego, czy obserwacja jest klasy 0, czy 1.
target = np.where((target == 0), 0, 1)

# Standaryzowanie cech.
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# Utworzenie klasyfikatora wektora nośnego.
svc = SVC(kernel="linear", class_weight="balanced", C=1.0, random_state=0)

# Wytrenowanie klasyfikatora.
model = svc.fit(features_standardized, target)
