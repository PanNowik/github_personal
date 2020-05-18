# Wczytanie bibliotek.
from keras import models
from keras import layers

# Uruchomienie sieci neuronowej.
network = models.Sequential()

# Dodanie w pełni połączonej warstwy z funkcją aktywacji ReLU.
network.add(layers.Dense(units=16, activation="relu", input_shape=(10,)))

# Dodanie w pełni połączonej warstwy z funkcją aktywacji ReLU.
network.add(layers.Dense(units=16, activation="relu"))

# Dodanie w pełni połączonej warstwy z esowatą funkcją aktywacji.
network.add(layers.Dense(units=1, activation="sigmoid"))

# Kompilacja sieci neuronowej.
network.compile(loss="binary_crossentropy", # Entropia krzyżowa.
                optimizer="rmsprop", # Propagacja RMS.
                metrics=["accuracy"]) # Zdefiniowanie dokładności jako współczynnka wydajności.
