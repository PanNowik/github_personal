# Wczytanie biblioteki.
import numpy as np

# Utworzenie macierzy.
matrix = np.array([[1, -1, 3],
                   [1, 1, 6],
                   [3, 8, 9]])

# Wyszukanie wektorów i wartości własnych.
eigenvalues, eigenvectors = np.linalg.eig(matrix)

# Wyświetlenie wartości własnych.
eigenvalues

array([ 13.55075847,   0.74003145,  -3.29078992])

# Wyświetlenie wektorów własnych.
eigenvectors
