# Wczytanie bibliotek.
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.datasets import make_regression

# Wygenerowanie macierzy cech i wektora docelowego.
features, target = make_regression(n_samples = 100,
                                   n_features = 3,
                                   random_state = 1)

# Utworzenie zbiorów uczącego i testowego.
features_train, features_test, target_train, target_test = train_test_split(
     features, target, test_size=0.10, random_state=1)

# Zdefiniowanie własnego współczynnika.
def custom_metric(target_test, target_predicted):
    # Obliczenie wartości R kwadrat.
    r2 = r2_score(target_test, target_predicted)
    # Zwrot wartości R kwadrat.
    return r2

# Utworzenie funkcji oceniającej i zdefiniowanie, że im większa wartość zwrotna, tym lepiej.
score = make_scorer(custom_metric, greater_is_better=True)

# Utworzenie obiektu regresji grzbietowej.
classifier = Ridge()

# Wytrenowanie modelu regresji grzbietowej.
model = classifier.fit(features_train, target_train)

# Zastosowanie własnej funkcji oceniającej.
score(model, features_test, target_test)
