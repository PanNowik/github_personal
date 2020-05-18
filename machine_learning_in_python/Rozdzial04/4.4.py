# Wczytanie bibliotek.
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# Utworzenie macierzy cech.
features = np.array([[2, 3],
                     [2, 3],
                     [2, 3]])

# Utworzenie obiektu typu PolynomialFeatures.
polynomial_interaction = PolynomialFeatures(degree=2, include_bias=False)

# Utworzenie cech wielomianowych.
polynomial_interaction.fit_transform(features)
