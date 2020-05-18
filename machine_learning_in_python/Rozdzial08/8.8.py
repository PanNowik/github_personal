# Wczytanie bibliotek.
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Wczytanie obrazu.
image_bgr = cv2.imread('images/plane_256x256.jpg')

# Konwersja przestrzeni barw BGR na HSV.
image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

# Zdefiniowanie zakresu koloru niebieskiego w HSV.
lower_blue = np.array([50,100,50])
upper_blue = np.array([130,255,255])

# Utworzenie maski.
mask = cv2.inRange(image_hsv, lower_blue, upper_blue)

# Nałożenie maski na obraz.
image_bgr_masked = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)

# Konwersja przestrzeni barw BGR na RGB.
image_rgb = cv2.cvtColor(image_bgr_masked, cv2.COLOR_BGR2RGB)

# Wyświetlenie obrazu.
plt.imshow(image_rgb), plt.axis("off")
plt.show()
