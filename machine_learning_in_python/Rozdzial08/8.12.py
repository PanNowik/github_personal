# Wczytanie bibliotek.
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Wczytanie obrazu jako czarno-białego.
image_bgr = cv2.imread("images/plane_256x256.jpg")
image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
image_gray = np.float32(image_gray)

# Zdefiniowanie parametrów wykrywacza narożników.
block_size = 2
aperture = 29
free_parameter = 0.04

# Wykrywanie narożników.
detector_responses = cv2.cornerHarris(image_gray,
                                      block_size,
                                      aperture,
                                      free_parameter)

# Oznaczenie dużych narożników.
detector_responses = cv2.dilate(detector_responses, None)

# Zachowane będą tylko dane większe niż podana wartość progowa i zostaną oznaczone kolorem białym.
threshold = 0.02
image_bgr[detector_responses >
          threshold *
          detector_responses.max()] = [255,255,255]

# Konwersja na obraz czarno-biały.
image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

# Wyświetlenie obrazu.
plt.imshow(image_gray, cmap="gray"), plt.axis("off")
plt.show()
