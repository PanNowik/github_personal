# Wczytanie bibliotek.
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Wczytanie obrazu i jego konwersja na RGB.
image_bgr = cv2.imread('images/plane_256x256.jpg')
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Wartości prostokątnego zaznaczenia: początkowy punkt na osi x, początkowy punkt na osi y, długość, wysokość.
rectangle = (0, 56, 256, 150)

# Utworzenie maski początkowej.
mask = np.zeros(image_rgb.shape[:2], np.uint8)

# Utworzenie tablic początkowych używanych przez algorytm GrabCut.
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# Wykonanie algorytmu GrabCut.
cv2.grabCut(image_rgb, # Obraz.
            mask, # Maska.
            rectangle, # Prostokąt.
            bgdModel, # Tablica tymczasowa dla tła.
            fgdModel, # Tablica tymczasowa dla pierwszego planu.
            5, # Liczba iteracji.
            cv2.GC_INIT_WITH_RECT) # Użycie argumentu GC_INIT_WITH_RECT.

# Utworzenie maski, w której piksele tła mają przypisaną wartość 0, natomiast piksele pozostałych elementów — wartość 1.
mask_2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')

# Nałożenie maski na obraz i odjęcie tła.
image_rgb_nobg = image_rgb * mask_2[:, :, np.newaxis]

# Wyświetlenie obrazu.
plt.imshow(image_rgb_nobg), plt.axis("off")
plt.show()
