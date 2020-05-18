# Wczytanie bibliotek.
import pandas as pd
import numpy as np

# Utworzenie macierzy cech z dwiema wysoce skorelowanymi cechami.
features = np.array([[1, 1, 1],
                     [2, 2, 0],
                     [3, 3, 1],
                     [4, 4, 0],
                     [5, 5, 1],
                     [6, 6, 0],
                     [7, 7, 1],
                     [8, 7, 0],
                     [9, 7, 1]])

# Konwersja macierzy cech na obiekt typu DataFrame.
dataframe = pd.DataFrame(features)

# Utworzenie macierzy korelacji.
corr_matrix = dataframe.corr().abs()

# Wybór górnego trójkąta macierzy korelacji.
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),
                          k=1).astype(np.bool))

# Wyszukanie indeksu kolumn cech o korelacji większej niż 0,95.
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

# Usunięcie cech.
dataframe.drop(dataframe.columns[to_drop], axis=1).head(3)
