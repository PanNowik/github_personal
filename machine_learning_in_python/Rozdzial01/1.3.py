# Wczytanie bibliotek.
import numpy as np
from scipy import sparse

# Utworzenie macierzy.
matrix = np.array([[0, 0],
                   [0, 1],
                   [3, 0]])

# Utworzenie macierzy w formacie CSR.
matrix_sparse = sparse.csr_matrix(matrix)
