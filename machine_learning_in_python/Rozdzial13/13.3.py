# Wczytanie biblioteki.
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.preprocessing import PolynomialFeatures

# Wczytanie danych zawierających jedną cechę.
boston = load_boston()
features = boston.data[:,0:1]
target = boston.target

# Utworzenie cech wielomianowych x^2 i x^3.
polynomial = PolynomialFeatures(degree=3, include_bias=False)
features_polynomial = polynomial.fit_transform(features)

# Zdefiniowanie regresji liniowej.
regression = LinearRegression()

# Wyznaczenie regresji liniowej.
model = regression.fit(features_polynomial, target)
