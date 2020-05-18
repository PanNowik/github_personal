# Wczytanie bibliotek.
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Wczytanie obrazu w przestrzeni barw BGR.
image_bgr = cv2.imread("images/plane_256x256.jpg", cv2.IMREAD_COLOR)

# Obliczenie średniej dla każdego kanału.
channels = cv2.mean(image_bgr)

# Zamiana miejscami wartości kolorów czerwonego i niebieskiego (zmiana na model barw RGB zamiast BGR).
observation = np.array([(channels[2], channels[1], channels[0])])

# Wyświetlenie wartości średniej dla poszczególnych kanałów.
observation
