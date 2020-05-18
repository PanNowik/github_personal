# Wczytanie bibliotek.
import numpy as np
from keras import models
from keras import layers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification

# Zdefiniowanie wartości zalążka.
np.random.seed(0)

# Liczba cech.
number_of_features = 100

# Wygenerowanie macierzy cech i wektora docelowego.
features, target = make_classification(n_samples = 10000,
                                       n_features = number_of_features,
                                       n_informative = 3,
                                       n_redundant = 0,
                                       n_classes = 2,
                                       weights = [.5, .5],
                                       random_state = 0)

# Utworzenie funkcji zwracającej skompilowaną sieć.
def create_network():

    # Uruchomienie sieci neuronowej.
    network = models.Sequential()

    # Dodanie w pełni połączonej warstwy z funkcją aktywacji ReLU.
    network.add(layers.Dense(units=16, activation="relu", input_shape=(
        number_of_features,)))

    # Dodanie w pełni połączonej warstwy z funkcją aktywacji ReLU.
    network.add(layers.Dense(units=16, activation="relu"))

    # Dodanie w pełni połączonej warstwy z esowatą funkcją aktywacji.
    network.add(layers.Dense(units=1, activation="sigmoid"))

    # Kompilacja sieci neuronowej.
    network.compile(loss="binary_crossentropy", # Entropia krzyżowa.
                    optimizer="rmsprop", # Propagacja RMS.
                    metrics=["accuracy"]) # Zdefiniowanie dokładności jako współczynnika wydajności.

    # Zwrot skompilowanej sieci.
    return network

# Opakowanie modelu Keras, co pozwala na jego użycie przez API biblioteki scikit-learn.
neural_network = KerasClassifier(build_fn=create_network,
                                 epochs=10,
                                 batch_size=100,
                                 verbose=0)

# Ocena sieci neuronowej za pomocą trzykrotnego sprawdzianu krzyżowego.
cross_val_score(neural_network, features, target, cv=3)
