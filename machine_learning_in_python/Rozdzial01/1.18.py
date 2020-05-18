# Wczytanie biblioteki.
import numpy as np

# Utworzenie macierzy.
matrix_a = np.array([[1, 1, 1],
                     [1, 1, 1],
                     [1, 1, 2]])

# Utworzenie macierzy.
matrix_b = np.array([[1, 3, 1],
                     [1, 3, 1],
                     [1, 3, 8]])

# Dodanie dwóch macierzy.
np.add(matrix_a, matrix_b)

array([[ 2,  4,  2],
       [ 2,  4,  2],
       [ 2,  4, 10]])

# Odjęcie jednej macierzy od drugiej.
np.subtract(matrix_a, matrix_b)
