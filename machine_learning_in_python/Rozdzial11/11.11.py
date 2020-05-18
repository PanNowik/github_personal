# Wczytanie bibliotek.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve

# Wczytanie danych.
digits = load_digits()

# Utworzenie macierzy cech i wektora docelowego.
features, target = digits.data, digits.target

# Zdefiniowanie sprawdzianu krzyżowego zbiorów uczącego i testowego dla ich różnych wielkości.
train_sizes, train_scores, test_scores = learning_curve(
    # Klasyfikator.
    RandomForestClassifier(),
    # Macierz cech.
    features,
    # Wektor docelowy.
    target,
    # Liczba podzbiorów.
    cv=10,
    # Metryka wydajności.
    scoring='accuracy',
    # Użycie wszystkich rdzeni procesora.
    n_jobs=-1,
    # Zbiór uczący ma wielkość 50.
    train_sizes=np.linspace(
    0.01,
    1.0,
    50))

# Obliczenie średniej i odchylenia standardowego dla wyników zbioru uczącego.
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Obliczenie średniej i odchylenia standardowego dla wyników zbioru testowego.
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Wyświetlenie linii.
plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Wynik uczenia")
plt.plot(train_sizes, test_mean, color="#111111", label="Wynik sprawdzianu krzyżowego")

# Wyświetlenie opaski.
plt.fill_between(train_sizes, train_mean - train_std,
                 train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std,
                 test_mean + test_std, color="#DDDDDD")

# Wyświetlenie wykresu.
plt.title("Krzywa uczenia")
plt.xlabel("Wielkość zbioru uczącego"), plt.ylabel("Dokładność"),
plt.legend(loc="best")
plt.tight_layout()
plt.show()
