# Wczytanie biblioteki.
import numpy as np

# Utworzenie macierzy o wielkości 4×3.
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9],
                   [10, 11, 12]])

# Zmiana kształtu macierzy na 2×6.
matrix.reshape(2, 6)
