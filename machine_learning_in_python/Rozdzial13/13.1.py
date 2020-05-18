# Wczytanie bibliotek.
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

# Wczytanie danych zawierających tylko dwie cechy.
boston = load_boston()
features = boston.data[:,0:2]
target = boston.target

# Zdefiniowanie regresji liniowej.
regression = LinearRegression()

# Wyznaczenie regresji liniowej.
model = regression.fit(features, target)
