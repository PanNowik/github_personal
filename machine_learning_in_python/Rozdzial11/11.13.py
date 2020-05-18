# Wczytanie bibliotek.
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import validation_curve

# Wczytanie danych.
digits = load_digits()

# Utworzenie macierzy cech i wektora docelowego.
features, target = digits.data, digits.target

# Zdefiniowanie zakresu wartości parametru.
param_range = np.arange(1, 250, 2)

# Obliczenie dokładności zbiorów uczącego i testowego dla zakresu wartości parametru.
train_scores, test_scores = validation_curve(
    # Klasyfikator.
    RandomForestClassifier(),
    # Macierz cech.
    features,
    # Wektor docelowy.
    target,
    # Analizowany hiperparametr.
    param_name="n_estimators",
    # Zakres wartości hiperparametru.
    param_range=param_range,
    # Liczba podzbiorów.
    cv=3,
    # Metryka wydajności.
    scoring="accuracy",
    # Użycie wszystkich rdzeni procesora.
    n_jobs=-1)

# Obliczenie średniej i odchylenia standardowego dla wyników zbioru uczącego.
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Obliczenie średniej i odchylenia standardowego dla wyników zbioru testowego.
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Wyświetlenie średniej wyników dokładności dla zbiorów uczącego i testowego.
plt.plot(param_range, train_mean, label="Wynik uczenia", color="black")
plt.plot(param_range, test_mean, label="Wynik sprawdzianu krzyżowego", color="dimgrey")

# Wygenerowanie wykresu dokładności dla zbiorów uczącego i testowego.
plt.fill_between(param_range, train_mean - train_std,
                 train_mean + train_std, color="gray")
plt.fill_between(param_range, test_mean - test_std,
                 test_mean + test_std, color="gainsboro")

# Wyświetlenie wykresu.
plt.title("Krzywa weryfikacji dla algorytmu losowego lasu")
plt.xlabel("Liczba drzew")
plt.ylabel("Dokładność")
plt.tight_layout()
plt.legend(loc="best")
plt.show()
