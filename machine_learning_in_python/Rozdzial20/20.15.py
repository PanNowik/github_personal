import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

# Zdefiniowanie, że wartość kanału koloru będzie pierwsza.
K.set_image_data_format("channels_first")

# Zdefiniowanie wartości zalążka.
np.random.seed(0)

# Zdefiniowanie informacji o obrazie.
channels = 1
height = 28
width = 28

# Wczytanie danych i wektora docelowego z danych MNIST.
(data_train, target_train), (data_test, target_test) = mnist.load_data()

# Przekształcenie danych uczących na cechy.
data_train = data_train.reshape(data_train.shape[0], channels, height, width)

# Przekształcenie danych testowych na cechy.
data_test = data_test.reshape(data_test.shape[0], channels, height, width)

# Zmiana skali intensywności piksela na wartość w przedziale od 0 do 1.
features_train = data_train / 255
features_test = data_test / 255

# Zastosowanie kodowania „gorącojedynkowego” dla wektora docelowego.
target_train = np_utils.to_categorical(target_train)
target_test = np_utils.to_categorical(target_test)
number_of_classes = target_test.shape[1]

# Uruchomienie sieci neuronowej.
network = Sequential()

# Dodanie warstwy splotowej wraz z 64 filtrami, oknem 5×5 i funkcją aktywacji ReLU.
network.add(Conv2D(filters=64,
                   kernel_size=(5, 5),
                   input_shape=(channels, width, height),
                   activation='relu'))

# Dodanie warstwy buforowej z oknem 2×2.
network.add(MaxPooling2D(pool_size=(2, 2)))

# Dodanie warstwy porzucenia jednostek.
network.add(Dropout(0.5))

# Dodanie warstwy spłaszczenia do danych wejściowych.
network.add(Flatten())

# Dodanie w pełni połączonej warstwy 128 jednostek z funkcją aktywacji ReLU.
network.add(Dense(128, activation="relu"))

# Dodanie warstwy porzucenia jednostek.
network.add(Dropout(0.5))

# Dodanie w pełni połączonej warstwy z funkcją aktywacji softmax.
network.add(Dense(number_of_classes, activation="softmax"))

# Kompilacja sieci neuronowej.
network.compile(loss="categorical_crossentropy", # Entropia krzyżowa.
                optimizer="rmsprop", # Propagacja RMS.
                metrics=["accuracy"]) # Zdefiniowanie dokładności jako współczynnika wydajności.

# Wytrenowanie sieci neuronowej.
network.fit(features_train, # Cechy.
            target_train, # Wektor docelowy.
            epochs=2, # Liczba epok.
            verbose=0, # Bez wyświetlania opisu po poszczególnych epokach.
            batch_size=1000, # Liczba obserwacji w zbiorze.
            validation_data=(features_test, target_test)) # Dane do oceny.
