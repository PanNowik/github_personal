# Wczytanie bibliotek.
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Wygenerowanie macierzy cech i wektora docelowego.
X, y = make_classification(n_samples = 10000,
                           n_features = 3,
                           n_informative = 3,
                           n_redundant = 0,
                           n_classes = 2,
                           random_state = 1)

# Utworzenie regresji logistycznej.
logit = LogisticRegression()

# Model sprawdzianu krzyżowego weryfikującego dokładność.
cross_val_score(logit, X, y, scoring="accuracy")
