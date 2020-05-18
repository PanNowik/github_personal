# Wczytanie obrazu.
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Wczytanie obrazu jako czarno-białego.
image = cv2.imread("images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# Zmiana wielkości obrazu na 10×10 pikseli.
image_10x10 = cv2.resize(image, (10, 10))

# Konwersja danych obrazu na wektor jednowymiarowy.
image_10x10.flatten()
