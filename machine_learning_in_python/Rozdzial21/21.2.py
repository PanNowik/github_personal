# Wczytanie bibliotek.
import numpy as np
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
from keras.models import load_model

# Zdefiniowanie wartości zalążka.
np.random.seed(0)

# Określenie żądanej liczby cech.
number_of_features = 1000

# Wczytanie danych i wektora docelowego z danych zawierających recenzje filmów.
(train_data, train_target), (test_data, test_target) = imdb.load_data(
    num_words=number_of_features)

# Konwersja danych recenzji filmów na macierz cech zakodowaną „gorącojedynkowo”.
tokenizer = Tokenizer(num_words=number_of_features)
train_features = tokenizer.sequences_to_matrix(train_data, mode="binary")
test_features = tokenizer.sequences_to_matrix(test_data, mode="binary")

# Uruchomienie sieci neuronowej.
network = models.Sequential()

# Dodanie w pełni połączonej warstwy z funkcją aktywacji ReLU.
network.add(layers.Dense(units=16,
                         activation="relu",
                         input_shape=(number_of_features,)))

# Dodanie w pełni połączonej warstwy z esowatą funkcją aktywacji.
network.add(layers.Dense(units=1, activation="sigmoid"))

# Kompilacja sieci neuronowej.
network.compile(loss="binary_crossentropy", # Entropia krzyżowa.
                optimizer="rmsprop", # Propagacja RMS.
                metrics=["accuracy"]) # Zdefiniowanie dokładności jako współczynnika wydajności.

# Wytrenowanie sieci neuronowej.
history = network.fit(train_features, # Cechy.
                      train_target, # Wektor docelowy.
                      epochs=3, # Liczba epok.
                      verbose=0, # Brak danych wyjściowych.
                      batch_size=100, # Liczba obserwacji w zbiorze.
                      validation_data=(test_features, test_target)) # Dane testowe.

# Zapisanie sieci neuronowej.
network.save("model.h5")
