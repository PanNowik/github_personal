# Wczytanie bibliotek.
from keras import models
from keras import layers
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

# Uruchomienie sieci neuronowej.
network = models.Sequential()

# Dodanie w pełni połączonej warstwy z funkcją aktywacji ReLU.
network.add(layers.Dense(units=16, activation="relu", input_shape=(10,)))

# Dodanie w pełni połączonej warstwy z funkcją aktywacji ReLU.
network.add(layers.Dense(units=16, activation="relu"))

# Dodanie w pełni połączonej warstwy z esowatą funkcją aktywacji.
network.add(layers.Dense(units=1, activation="sigmoid"))

# Wizualizacja architektury sieci.
SVG(model_to_dot(network, show_shapes=True).create(prog="dot", format="svg"))
