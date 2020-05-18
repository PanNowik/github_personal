# Wczytanie bibliotek.
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Wczytanie obrazu.
image_bgr = cv2.imread("images/plane_256x256.jpg", cv2.IMREAD_COLOR)

# Konwersja na przestrzeń barw RGB.
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Utworzenie listy dla wartości cech.
features = []

# Wygenerowanie histogramu dla każdego kanału koloru.
colors = ("r","g","b")

# Dla każdego kanału: wygenerowanie histogramu i dodanie go do listy wartości cech.
for i, channel in enumerate(colors):
    histogram = cv2.calcHist([image_rgb], # Obraz.
                        [i], # Indeks kanału.
                        None, # Brak maski.
                        [256], # Wielkość histogramu.
                        [0,256]) # Zakres wartości.
    features.extend(histogram)

# Utworzenie wektora dla wartości cech obserwacji.
observation = np.array(features).flatten()

# Wyświetlenie wartości obserwacji dla pierwszych pięciu cech.
observation[0:5]
