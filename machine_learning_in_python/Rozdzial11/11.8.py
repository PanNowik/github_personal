# Wczytanie bibliotek.
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

# Wygenerowanie macierzy cech i wektora docelowego.
features, target = make_regression(n_samples = 100,
                                   n_features = 3,
                                   n_informative = 3,
                                   n_targets = 1,
                                   noise = 50,
                                   coef = False,
                                   random_state = 1)

# Utworzenie obiektu regresji liniowej.
ols = LinearRegression()

# Sprawdzian krzyżowy regresji liniowej za pomocą (negatywnego) MSE.
cross_val_score(ols, features, target, scoring='neg_mean_squared_error')
