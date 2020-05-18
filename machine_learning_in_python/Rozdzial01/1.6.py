# Wczytanie biblioteki.
import numpy as np

# Utworzenie macierzy.
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Utworzenie funkcji dodającej 100 do pewnej wartości.
add_100 = lambda i: i + 100

# Utworzenie funkcji klasy vectorize.
vectorized_add_100 = np.vectorize(add_100)

# Wywołanie funkcji dla wszystkich elementów tablicy.
vectorized_add_100(matrix)
