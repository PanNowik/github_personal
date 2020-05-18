# Wczytanie bibliotek.
import numpy as np
from keras import models
from keras import layers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
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
def create_network(optimizer="rmsprop"):

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
                    optimizer=optimizer, # Metoda optymalizacji.
                    metrics=["accuracy"]) # Zdefiniowanie dokładności jako współczynnika wydajności.

    # Zwrot skompilowanej sieci.
    return network

# Opakowanie modelu Keras, co pozwala na jego użycie przez API biblioteki scikit-learn.
neural_network = KerasClassifier(build_fn=create_network, verbose=0)

# Zdefiniowanie przestrzeni hiperparametru.
epochs = [5, 10]
batches = [5, 10, 100]
optimizers = ["rmsprop", "adam"]

# Zdefiniowanie opcji hiperparametru.
hyperparameters = dict(optimizer=optimizers, epochs=epochs, batch_size=batches)

# Zdefiniowanie przeszukiwania siatki.
grid = GridSearchCV(estimator=neural_network, param_grid=hyperparameters)

# Dopasowanie przeszukiwania siatki.
grid_result = grid.fit(features, target)
