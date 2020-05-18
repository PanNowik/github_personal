# Wczytanie biblioteki.
import numpy as np

# Utworzenie macierzy.
matrix = np.array([[1, 1, 1],
                   [1, 1, 10],
                   [1, 1, 15]])

# Zwrot rzędu macierzy.
np.linalg.matrix_rank(matrix)
