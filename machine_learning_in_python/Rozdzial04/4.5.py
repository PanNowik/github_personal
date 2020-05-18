# Wczytanie bibliotek.
import numpy as np
from sklearn.preprocessing import FunctionTransformer

# Utworzenie macierzy cech.
features = np.array([[2, 3],
                     [2, 3],
                     [2, 3]])

# Zdefiniowanie prostej funkcji.
def add_ten(x):
    return x + 10

# Przygotowanie operacji przekształcenia.
ten_transformer = FunctionTransformer(add_ten)

# Przekształcenie macierzy cech.
ten_transformer.transform(features)
