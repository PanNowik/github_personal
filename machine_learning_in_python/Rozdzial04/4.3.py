# Wczytanie bibliotek.
import numpy as np
from sklearn.preprocessing import Normalizer

# Utworzenie macierzy cech.
features = np.array([[0.5, 0.5],
                     [1.1, 3.4],
                     [1.5, 20.2],
                     [1.63, 34.4],
                     [10.9, 3.3]])

# Zdefiniowanie normalizacji.
normalizer = Normalizer(norm="l2")

# Przekształcenie macierzy cech.
normalizer.transform(features)
