# Wczytanie bibliotek.
import numpy as np
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
from keras import regularizers

# Zdefiniowanie wartości zalążka.
np.random.seed(0)

# Określenie żądanej liczby cech.
number_of_features = 1000

# Wczytanie danych i wektora docelowego z danych zawierających recenzje filmów.
(data_train, target_train), (data_test, target_test) = imdb.load_data(
    num_words=number_of_features)

# Konwersja danych recenzji filmów na macierz cech zakodowaną „gorącojedynkowo”.
tokenizer = Tokenizer(num_words=number_of_features)
features_train = tokenizer.sequences_to_matrix(data_train, mode="binary")
features_test = tokenizer.sequences_to_matrix(data_test, mode="binary")

# Uruchomienie sieci neuronowej.
network = models.Sequential()

# Dodanie w pełni połączonej warstwy z funkcją aktywacji ReLU.
network.add(layers.Dense(units=16,
                         activation="relu",
                         kernel_regularizer=regularizers.l2(0.01),
                         input_shape=(number_of_features,)))

# Dodanie w pełni połączonej warstwy z funkcją aktywacji ReLU.
network.add(layers.Dense(units=16,
                         kernel_regularizer=regularizers.l2(0.01),
                         activation="relu"))

# Dodanie w pełni połączonej warstwy z esowatą funkcją aktywacji.
network.add(layers.Dense(units=1, activation="sigmoid"))

# Kompilacja sieci neuronowej.
network.compile(loss="binary_crossentropy", # Entropia krzyżowa.
                optimizer="rmsprop", # Propagacja RMS.
                metrics=["accuracy"]) # Zdefiniowanie dokładności jako współczynnika wydajności.

# Wytrenowanie sieci neuronowej.
history = network.fit(features_train, # Cechy.
                      target_train, # Wektor docelowy.
                      epochs=3, # Liczba epok.
                      verbose=0, # Brak danych wyjściowych.
                      batch_size=100, # Liczba obserwacji w zbiorze.
                      validation_data=(features_test, target_test)) # Dane testowe.
