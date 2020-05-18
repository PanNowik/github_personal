# Wczytanie biblioteki.
from sklearn.datasets import make_regression

# Wygenerowanie macierzy cech, wektora docelowego i prawdziwych współczynników.
features, target, coefficients = make_regression(n_samples = 100,
                                                 n_features = 3,
                                                 n_informative = 3,
                                                 n_targets = 1,
                                                 noise = 0.0,
                                                 coef = True,
                                                 random_state = 1)

# Wyświetlenie macierzy cech i wektora docelowego.
print('Macierz cech\n', features[:3])
print('Wektor docelowy\n', target[:3])
