# Wczytanie biblioteki.
import numpy as np

# Utworzenie macierzy.
matrix = np.array([[1, 2, 3],
                   [2, 4, 6],
                   [3, 8, 9]])

# Zwrot wyznacznika macierzy.
np.linalg.det(matrix)
