# Wczytanie bibliotek.
import warnings
from sklearn.datasets import make_regression
from sklearn.feature_selection import RFECV
from sklearn import datasets, linear_model

# Pozbycie się irytującego, choć nieszkodliwego ostrzeżenia.
warnings.filterwarnings(action="ignore", module="scipy",
                        message="^internal gelsd")

# Wygenerowanie macierzy cech, wektora docelowego i prawdziwych współczynników.
features, target = make_regression(n_samples = 10000,
                                   n_features = 100,
                                   n_informative = 2,
                                   random_state = 1)

# Zdefiniowanie regresji liniowej.
ols = linear_model.LinearRegression()

# Rekurencyjne eliminowanie cech.
rfecv = RFECV(estimator=ols, step=1, scoring="neg_mean_squared_error")
rfecv.fit(features, target)
rfecv.transform(features)
