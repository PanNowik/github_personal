# Wczytanie bibliotek.
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Zdefiniowanie wartości zalążka.
np.random.seed(0)

# Wygenerowanie macierzy cech i wektora docelowego.
features, target = make_regression(n_samples = 10000,
                                   n_features = 3,
                                   n_informative = 3,
                                   n_targets = 1,
                                   noise = 0.0,
                                   random_state = 0)

# Podział danych na zbiory uczący i testowy.
features_train, features_test, target_train, target_test = train_test_split(
features, target, test_size=0.33, random_state=0)

# Uruchomienie sieci neuronowej.
network = models.Sequential()

# Dodanie w pełni połączonej warstwy z funkcją aktywacji ReLU.
network.add(layers.Dense(units=32,
                         activation="relu",
                         input_shape=(features_train.shape[1],)))

# Dodanie w pełni połączonej warstwy z funkcją aktywacji ReLU.
network.add(layers.Dense(units=32, activation="relu"))

# Dodanie w pełni połączonej warstwy bez funkcji aktywacji.
network.add(layers.Dense(units=1))

# Kompilacja sieci neuronowej.
network.compile(loss="mse", # Błąd średniokwadratowy.
                optimizer="RMSprop", # Algorytm optymalizacji.
                metrics=["mse"]) # Błąd średniokwadratowy.

# Wytrenowanie sieci neuronowej.
history = network.fit(features_train, # Cechy.
                      target_train, # Wektor docelowy.
                      epochs=10, # Liczba epok.
                      verbose=0, # Brak danych wyjściowych.
                      batch_size=100, # Liczba obserwacji w zbiorze.
                      validation_data=(features_test, target_test)) # Dane testowe.
