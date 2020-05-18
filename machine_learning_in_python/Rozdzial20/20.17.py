# Wczytanie bibliotek.
import numpy as np
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import models
from keras import layers

# Zdefiniowanie wartości zalążka.
np.random.seed(0)

# Określenie żądanej liczby cech.
number_of_features = 1000

# Wczytanie danych i wektora docelowego z danych zawierających recenzje filmów.
(data_train, target_train), (data_test, target_test) = imdb.load_data(
    num_words=number_of_features)

# Zastosowanie dopełnienia lub skrócenia, aby każda obserwacja składała się z 400 cech.
features_train = sequence.pad_sequences(data_train, maxlen=400)
features_test = sequence.pad_sequences(data_test, maxlen=400)

# Uruchomienie sieci neuronowej.
network = models.Sequential()

# Dodanie warstwy typu Embedding.
network.add(layers.Embedding(input_dim=number_of_features, output_dim=128))

# Dodanie warstwy typu LSTM wraz ze 128 jednostkami.
network.add(layers.LSTM(units=128))

# Dodanie w pełni połączonej warstwy z esowatą funkcją aktywacji.
network.add(layers.Dense(units=1, activation="sigmoid"))

# Kompilacja sieci neuronowej.
network.compile(loss="binary_crossentropy", # Entropia krzyżowa.
                optimizer="Adam", # Optymalizacja Adam.
                metrics=["accuracy"]) # Zdefiniowanie dokładności jako współczynnika wydajności.

# Wytrenowanie sieci neuronowej.
history = network.fit(features_train, # Cechy.
                      target_train, # Wektor docelowy.
                      epochs=3, # Liczba epok.
                      verbose=0, # Brak wyświetlania informacji po każdej epoce.
                      batch_size=1000, # Liczba obserwacji w zbiorze.
                      validation_data=(features_test, target_test)) # Dane testowe.
