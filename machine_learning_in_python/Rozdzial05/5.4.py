# Wczytanie bibliotek.
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Utworzenie macierzy cech wraz z cechą kategoryzującą.
X = np.array([[0, 2.10, 1.45],
              [1, 1.18, 1.33],
              [0, 1.22, 1.27],
              [1, -0.21, -1.19]])

# Utworzenie macierzy cechy wraz z brakującymi wartościami cechy kategoryzującej.
X_with_nan = np.array([[np.nan, 0.87, 1.31],
                       [np.nan, -0.67, -0.22]])

# Wytrenowanie algorytmu KNN.
clf = KNeighborsClassifier(3, weights='distance')
trained_model = clf.fit(X[:,1:], X[:,0])

# Prognozowanie brakujących wartości klas.
imputed_values = trained_model.predict(X_with_nan[:,1:])

# Połączenie kolumny prognozowanej klasy wraz z pozostałymi cechami.
X_with_imputed = np.hstack((imputed_values.reshape(-1,1), X_with_nan[:,1:]))

# Połączenie dwóch wskaźników cech.
np.vstack((X_with_imputed, X))
