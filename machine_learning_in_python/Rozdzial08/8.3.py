# Wczytanie bibliotek.
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Wczytanie obrazu jako czarno-białego.
image = cv2.imread("images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# Zmiana wielkości obrazu na 50×50 pikseli.
image_50x50 = cv2.resize(image, (50, 50))

# Wyświetlenie obrazu.
plt.imshow(image_50x50, cmap="gray"), plt.axis("off")
plt.show()
