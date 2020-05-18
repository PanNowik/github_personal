# Wczytanie bibliotek.
import numpy as np
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint

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
                         input_shape=(number_of_features,)))

# Dodanie w pełni połączonej warstwy z funkcją aktywacji ReLU.
network.add(layers.Dense(units=16, activation="relu"))

# Dodanie w pełni połączonej warstwy z esowatą funkcją aktywacji.
network.add(layers.Dense(units=1, activation="sigmoid"))

# Kompilacja sieci neuronowej.
network.compile(loss="binary_crossentropy", # Entropia krzyżowa.
                optimizer="rmsprop", # Propagacja RMS.
                metrics=["accuracy"]) # Zdefiniowanie dokładności jako współczynnika wydajności.

# Zdefiniowanie funkcji wywołania zwrotnego, aby wcześniej zakończyć proces uczenia i zachować najlepszy dotąd model.
callbacks = [EarlyStopping(monitor="val_loss", patience=2),
             ModelCheckpoint(filepath="best_model.h5",
                             monitor="val_loss",
                             save_best_only=True)]

# Wytrenowanie sieci neuronowej.
history = network.fit(features_train, # Cechy.
                      target_train, # Wektor docelowy.
                      epochs=20, # Liczba epok.
                      callbacks=callbacks, # Wcześniejsze zakończenie procesu uczenia.
                      verbose=0, # Wyświetlenie opisu po każdej epoce.
                      batch_size=100, # Liczba obserwacji w zbiorze.
                      validation_data=(features_test, target_test)) # Dane testowe.
