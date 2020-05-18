# Wczytanie bibliotek.
import numpy as np
from keras.datasets import reuters
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers

# Zdefiniowanie wartości zalążka.
np.random.seed(0)

# Określenie żądanej liczby cech.
number_of_features = 5000

# Wczytanie cech i danych docelowych.
data = reuters.load_data(num_words=number_of_features)
(data_train, target_vector_train), (data_test, target_vector_test) = data

# Konwersja danych cech na macierz cech zakodowaną „gorącojedynkowo”.
tokenizer = Tokenizer(num_words=number_of_features)
features_train = tokenizer.sequences_to_matrix(data_train, mode="binary")
features_test = tokenizer.sequences_to_matrix(data_test, mode="binary")

# Zakodowany „gorącojedynkowo” wektor docelowy przeznaczony do utworzenia macierzy docelowej.
target_train = to_categorical(target_vector_train)
target_test = to_categorical(target_vector_test)

# Uruchomienie sieci neuronowej.
network = models.Sequential()

# Dodanie w pełni połączonej warstwy z funkcją aktywacji ReLU.
network.add(layers.Dense(units=100,
                         activation="relu",
                         input_shape=(number_of_features,)))

# Dodanie w pełni połączonej warstwy z funkcją aktywacji ReLU.
network.add(layers.Dense(units=100, activation="relu"))

# Dodanie w pełni połączonej warstwy z funkcją aktywacji softmax.
network.add(layers.Dense(units=46, activation="softmax"))

# Kompilacja sieci neuronowej.
network.compile(loss="categorical_crossentropy", # Entropia krzyżowa.
                optimizer="rmsprop", # Propagacja RMS.
                metrics=["accuracy"]) # Zdefiniowanie dokładności jako współczynnika wydajności.

# Wytrenowanie sieci neuronowej.
history = network.fit(features_train, # Cechy.
                      target_train, # Wektor docelowy.
                      epochs=3, # Trzy epoki.
                      verbose=0, # Brak danych wyjściowych.
                      batch_size=100, # Liczba obserwacji w zbiorze.
                      validation_data=(features_test, target_test)) # Dane testowe.
