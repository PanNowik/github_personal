# Wczytanie bibliotek.
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.preprocessing import PolynomialFeatures

# Wczytanie danych zawierających tylko dwie cechy.
boston = load_boston()
features = boston.data[:,0:2]
target = boston.target

# Utworzenie pewnej interakcji.
interaction = PolynomialFeatures(
    degree=3, include_bias=False, interaction_only=True)
features_interaction = interaction.fit_transform(features)

# Zdefiniowanie regresji liniowej.
regression = LinearRegression()

# Wyznaczenie regresji liniowej.
model = regression.fit(features_interaction, target)
