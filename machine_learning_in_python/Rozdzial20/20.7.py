# Wczytanie bibliotek.
import numpy as np
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
import matplotlib.pyplot as plt

# Zdefiniowanie wartości zalążka.
np.random.seed(0)

# Określenie żądanej liczby cech.
number_of_features = 10000

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

# Wytrenowanie sieci neuronowej.
history = network.fit(features_train, # Cechy.
                      target_train, # Wektor docelowy.
                      epochs=15, # Liczba epok.
                      verbose=0, # Brak danych wyjściowych.
                      batch_size=1000, # Liczba obserwacji w zbiorze.
                      validation_data=(features_test, target_test)) # Dane testowe.

# Pobranie historii straty zbiorów uczącego i testowego.
training_loss = history.history["loss"]
test_loss = history.history["val_loss"]

# Określenie liczby epok.
epoch_count = range(1, len(training_loss) + 1)

# Wizualizacja historii straty.
plt.plot(epoch_count, training_loss, "r--")
plt.plot(epoch_count, test_loss, "b-")
plt.legend(["Strata zbioru uczącego", "Strata zbioru testowego"])
plt.xlabel("Epoka")
plt.ylabel("Strata")
plt.show()
