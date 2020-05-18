# Wczytanie bibliotek.
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Wczytanie danych Iris.
iris = load_iris()

# Utworzenie macierzy cech.
features = iris.data

# Utworzenie wektora docelowego.
target = iris.target

# Usunięcie pierwszych 40 obserwacji.
features = features[40:,:]
target = target[40:]

# Utworzenie binarnego wektora docelowego wskazującego, czy mamy do czynienia z klasą 0.
target = np.where((target == 0), 0, 1)

# Wyświetlenie niezrównoważonego wektora docelowego.
target
